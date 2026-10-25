"""Unified profiles (spec: docs/specs/voice-studio-unification.md §3/§5).

Endpoint validation tests run against an isolated tmp data dir; the design
render path is monkeypatched (no model in CI). Migration tests drive alembic
programmatically. Patterns mirror tests/test_profile_consent.py.
"""

import json
import os
import sqlite3

import pytest

os.environ.setdefault("OMNIVOICE_MODEL", "test")
os.environ.setdefault("OMNIVOICE_DISABLE_FILE_LOG", "1")

_FAKE_AUDIO = b"RIFF" + b"\x00" * 2000
_VD = {"Gender": "female", "Age": "young adult", "Pitch": "high pitch"}


@pytest.fixture(scope="module")
def app_client(tmp_path_factory):
    """TestClient with an isolated data dir (no lifespan — schema only)."""
    mp = pytest.MonkeyPatch()
    tmp_path = tmp_path_factory.mktemp("unified-profiles-data")
    mp.setenv("OMNIVOICE_DATA_DIR", str(tmp_path))

    import importlib
    import core.config as _cfg
    importlib.reload(_cfg)
    import core.db as _db
    importlib.reload(_db)
    from api.routers import profiles as _profiles
    importlib.reload(_profiles)
    import main as _main
    importlib.reload(_main)

    _db.init_db()

    from fastapi.testclient import TestClient
    try:
        yield TestClient(_main.app, client=("127.0.0.1", 50001)), _cfg
    finally:
        mp.undo()


@pytest.fixture()
def fake_render(monkeypatch):
    """Stub the design sample renderer — CI has no TTS engine."""
    async def _fake(a, out_path, **kwargs):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(_FAKE_AUDIO)

    from api.routers import archetypes as _arch
    from services import model_manager
    monkeypatch.setattr(model_manager, "get_model_status", lambda: {"loaded": True})
    monkeypatch.setattr(_arch, "_render_archetype_wav", _fake)
    return _fake


# ── Create validation ────────────────────────────────────────────────────────

def test_clone_requires_ref_audio(app_client):
    client, _ = app_client
    r = client.post("/profiles", data={"name": "NoAudio"})
    assert r.status_code == 422


def test_design_requires_vd_states(app_client):
    client, _ = app_client
    r = client.post("/profiles", data={"name": "D", "kind": "design", "instruct": "female, calm"})
    assert r.status_code == 422


def test_design_saveable_without_instruct(app_client, fake_render):
    """An all-Auto design (no picks → empty instruct) is still saveable (#476).

    Saving must not gate on a non-empty instruct — synthesis falls back to
    neutral instruct-only conditioning.
    """
    client, _ = app_client
    all_auto = {"Gender": "Auto", "Age": "Auto", "Pitch": "Auto"}
    r = client.post(
        "/profiles",
        data={"name": "AllAuto", "kind": "design", "vd_states": json.dumps(all_auto)},
    )
    assert r.status_code == 200, r.text
    profile = client.get(f"/profiles/{r.json()['id']}").json()
    assert profile["kind"] == "design"
    assert (profile["instruct"] or "") == ""


def test_design_derives_instruct_from_picks_when_unset(app_client, fake_render):
    """A design saved with category picks but no explicit instruct must persist
    a matching instruct derived from vd_states — otherwise the designed voice
    renders neutral/wrong-gender (#594). Guards the save-time heal."""
    client, _ = app_client
    r = client.post(
        "/profiles",
        data={"name": "Designed", "kind": "design", "vd_states": json.dumps(_VD)},
    )
    assert r.status_code == 200, r.text
    profile = client.get(f"/profiles/{r.json()['id']}").json()
    assert profile["instruct"] == "female, young adult, high pitch"


def test_design_save_drops_object_object_instruct(app_client, fake_render):
    """The "[object Object]" poison can never be persisted — it's sanitized and
    the design is rebuilt from vd_states at save time (#550 #571 #594)."""
    client, _ = app_client
    r = client.post(
        "/profiles",
        data={
            "name": "Poisoned", "kind": "design",
            "vd_states": json.dumps(_VD), "instruct": "[object Object]",
        },
    )
    assert r.status_code == 200, r.text
    profile = client.get(f"/profiles/{r.json()['id']}").json()
    assert profile["instruct"] == "female, young adult, high pitch"


def test_native_voice_design_profile_keeps_full_description(app_client):
    """Saving a VoxCPM2 voice cannot silently collapse its prose to tags."""
    from api.routers.generation import _resolve_profile_conditioning
    from core.db import db_conn
    from services.voice_design_profiles import profile_instruct

    client, _ = app_client
    raw = {
        "description": "A soft-spoken Japanese narrator, a little breathy and introspective.",
        "picks": {"Pitch": "low pitch"},
    }
    r = client.post("/profiles", data={
        "name": "Native Designer", "kind": "design", "engine": "voxcpm2",
        "vd_states": json.dumps(_VD), "design_recipe": json.dumps(raw),
    })
    assert r.status_code == 200, r.text
    profile = r.json()
    assert profile["ref_audio_path"] is None  # no accidental OmniVoice rendering
    assert profile["instruct"] == "female, young adult, high pitch"
    stored = json.loads(profile["design_recipe"])
    assert stored["description"] == raw["description"]
    assert stored["engine"] == "voxcpm2"
    assert stored["picks"] == raw["picks"]

    with db_conn() as conn:
        row = conn.execute("SELECT * FROM voice_profiles WHERE id=?", (profile["id"],)).fetchone()
    assert profile_instruct(row, "omnivoice") == "female, young adult, high pitch"
    assert profile_instruct(row, "voxcpm2") == raw["description"] + ", low pitch"
    cond = _resolve_profile_conditioning(row, engine_id="voxcpm2")
    assert cond["instruct"] == raw["description"] + ", low pitch"
    assert cond["ref_audio_path"] is None
    # The Design workspace sends this same instruction; it must not be
    # mistaken for a different identity and clear the saved profile link.
    assert not _resolve_profile_conditioning(
        row, engine_id="voxcpm2", instruct=cond["instruct"], seed=profile["seed"]
    )["diverged"]


def test_native_profile_preview_uses_its_authoring_engine(app_client, monkeypatch):
    """A saved VoxCPM2 voice must not silently render an OmniVoice sample."""
    client, _ = app_client
    import api.routers.profiles as profiles_router

    recipe = {"description": "a soft warm narrator", "picks": {}}
    created = client.post("/profiles", data={
        "name": "Native Preview", "kind": "design", "engine": "voxcpm2",
        "vd_states": json.dumps(_VD), "design_recipe": json.dumps(recipe),
    })
    assert created.status_code == 200, created.text
    profile_id = created.json()["id"]
    called = []

    async def fake_render(row, path, engine_id):
        """Write a stub identity WAV while recording the chosen model."""
        import wave
        called.append((row["design_recipe"], engine_id))
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\\x01\\x00" * 24000)

    monkeypatch.setattr(profiles_router, "_render_native_design_sample", fake_render)
    audio = client.get(f"/profiles/{profile_id}/audio")
    assert audio.status_code == 200, audio.text[:300] if audio.status_code != 200 else ""
    assert audio.headers["content-type"].startswith("audio/")
    assert len(called) == 1 and called[0][1] == "voxcpm2"
    assert json.loads(called[0][0])["description"] == recipe["description"]
    profile = client.get(f"/profiles/{profile_id}").json()
    assert profile["ref_audio_path"] == profile_id + ".wav"
    assert client.get(f"/profiles/{profile_id}/audio").status_code == 200
    assert len(called) == 1


@pytest.mark.asyncio
async def test_native_preview_reuses_engine_cache_and_sends_seed(tmp_path, monkeypatch):
    """Pending samples use the shared engine and forward seeds to sidecars."""
    import torch
    from api.routers import engines as engines_router
    from api.routers import generation as generation_router
    from api.routers import profiles as profiles_router
    from services import engine_memory, gpu_gateway, tts_backend, watermark

    class StubBackend:
        """An in-memory engine with a real WAV tensor and seed-aware protocol."""
        id = "voxcpm2"
        supports_generation_seed = True
        instruct_vocabulary = "freeform"
        sample_rate = 24000

        @classmethod
        def is_available(cls):
            """Pretend a previously installed sidecar is healthy."""
            return True, "ready"

        def generate(self, text, **kw):
            """Record the actual kwargs a sidecar would receive."""
            calls.append(kw)
            return torch.ones(1, 24000) * 0.1

    calls = []
    engine = StubBackend()
    cached = []
    prewarmed = []
    evicted = []

    def get_cached(cls):
        """Return the same adapter instance across independent previews."""
        cached.append(cls)
        return engine

    async def prewarm(*, backend, engine, **kw):
        """Record model readiness without starting GPU workers."""
        prewarmed.append((backend, engine))

    async def evict(engine_id):
        """Track the single-active-engine eviction request."""
        evicted.append(engine_id)

    async def run_guarded(fn, **kw):
        """Execute stub synthesis instead of dispatching to the real GPU."""
        return fn()

    async def passthrough(wav, sample_rate, **kw):
        """Skip provenance cost while preserving the generated tensor."""
        return wav

    monkeypatch.setattr(tts_backend, "get_backend_class", lambda name: StubBackend)
    monkeypatch.setattr(engines_router, "_get_engine_instance", get_cached)
    monkeypatch.setattr(gpu_gateway, "prewarm", prewarm)
    monkeypatch.setattr(engine_memory, "evict_other_tts_engines", evict)
    monkeypatch.setattr(generation_router, "run_on_gpu_pool_guarded", run_guarded)
    monkeypatch.setattr(watermark, "mark_synthetic_async", passthrough)

    row = {
        "kind": "design", "ref_text": "Speak softly.", "language": "Auto",
        "instruct": "female", "vd_states": None, "seed": 123,
        "design_recipe": json.dumps({
            "description": "Warm and gentle narration.", "picks": {},
            "engine": "voxcpm2",
        }),
    }
    for i in range(2):
        await profiles_router._render_native_design_sample(
            row, str(tmp_path / f"sample-{i}.wav"), "voxcpm2",
        )
        assert (tmp_path / f"sample-{i}.wav").is_file()
    assert cached == [StubBackend, StubBackend]
    assert prewarmed == [(engine, "voxcpm2")] * 2
    assert evicted == ["voxcpm2", "voxcpm2"]
    assert len(calls) == 2
    assert all(call["seed"] == 123 for call in calls)
    assert all(call["instruct"] == "Warm and gentle narration." for call in calls)


def test_editing_native_design_invalidates_stale_recipe_and_reference(app_client):
    """Legacy tag edits intentionally replace stale prose and sample identity."""
    from core.db import db_conn

    client, _ = app_client
    recipe = {
        "description": "A warm Japanese narrator, slightly breathy and restrained.",
        "picks": {"Age": "young adult"},
    }
    response = client.post("/profiles", data={
        "name": "Edited native design", "kind": "design", "engine": "voxcpm2",
        "vd_states": json.dumps(_VD), "design_recipe": json.dumps(recipe),
    })
    assert response.status_code == 200, response.text
    pid = response.json()["id"]
    assert response.json()["design_recipe"]

    # Changing only the profile name or saving the same tags preserves recipe.
    unchanged = client.put(f"/profiles/{pid}", json={"name": "New name"})
    assert unchanged.status_code == 200
    assert json.loads(unchanged.json()["design_recipe"])["description"] == recipe["description"]
    tags = unchanged.json()["instruct"]
    unchanged_tags = client.put(f"/profiles/{pid}", json={"instruct": tags})
    assert unchanged_tags.status_code == 200
    assert unchanged_tags.json()["design_recipe"]

    # An actual tag change must not secretly reapply the old free-form prose.
    with db_conn() as conn:
        conn.execute("UPDATE voice_profiles SET ref_audio_path=? WHERE id=?", (
            f"{pid}.wav", pid,
        ))
    changed = client.put(f"/profiles/{pid}", json={"instruct": "female, low pitch"})
    assert changed.status_code == 200, changed.text
    assert changed.json()["instruct"] == "female, low pitch"
    assert changed.json()["design_recipe"] is None
    assert changed.json()["ref_audio_path"] is None
    assert json.loads(changed.json()["vd_states"])["Pitch"] == "low pitch"


def test_native_profile_rejects_invalid_recipe_and_reference_only_engine(app_client):
    """Disallow malformed freeform inputs and reference-only design engines."""
    client, _ = app_client
    data = {
        "name": "Invalid", "kind": "design", "engine": "voxcpm2",
        "vd_states": json.dumps(_VD),
    }
    assert client.post("/profiles", data={**data, "design_recipe": "{broken"}).status_code == 422
    assert client.post("/profiles", data={
        **data, "engine": "indextts2",
        "design_recipe": json.dumps({"description": "low smoky voice", "picks": {}}),
    }).status_code == 422


def test_design_rejects_malformed_vd_states(app_client):
    client, _ = app_client
    r = client.post(
        "/profiles",
        data={"name": "D", "kind": "design", "vd_states": "[1,2]", "instruct": "female"},
    )
    assert r.status_code == 422


def test_unknown_kind_rejected(app_client):
    client, _ = app_client
    r = client.post(
        "/profiles",
        data={"name": "X", "kind": "mystery"},
        files={"ref_audio": ("x.wav", _FAKE_AUDIO, "audio/wav")},
    )
    assert r.status_code == 422


# ── Create happy paths ───────────────────────────────────────────────────────

def test_clone_create_defaults_kind(app_client):
    client, _ = app_client
    r = client.post(
        "/profiles",
        data={"name": "Clone Me"},
        files={"ref_audio": ("me.wav", _FAKE_AUDIO, "audio/wav")},
    )
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    profile = client.get(f"/profiles/{pid}").json()
    assert profile["kind"] == "clone"
    assert profile["vd_states"] is None


def test_design_create_renders_sample_and_stores_params(app_client, fake_render):
    client, cfg = app_client
    r = client.post(
        "/profiles",
        data={
            "name": "Designed",
            "kind": "design",
            "vd_states": json.dumps(_VD),
            "instruct": "female, young adult, high pitch",
            "language": "English",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "design"
    profile = client.get(f"/profiles/{body['id']}").json()
    assert profile["kind"] == "design"
    # #983: the server now completes vd_states to all 6 known categories
    # (missing ones default to 'Auto') before persisting — _VD only sets 3,
    # so the stored value is a superset of it, not an exact match.
    stored = json.loads(profile["vd_states"])
    assert stored == {**_VD, "Style": "Auto", "EnglishAccent": "Auto", "ChineseDialect": "Auto"}
    assert profile["seed"] == 42  # deterministic identity sample
    wav = os.path.join(cfg.VOICES_DIR, profile["ref_audio_path"])
    assert os.path.exists(wav) and os.path.getsize(wav) > 0


def test_design_normalizes_partial_vd_states_to_all_categories(app_client, fake_render):
    """#983: a design profile must never persist with a partial vd_states shape.

    A client (older frontend build, hand-edited payload, third-party API
    caller) that only sends a subset of the 6 known category keys used to be
    saved as-is — selecting that profile later handed the frontend an
    incomplete vdStates object, crashing DesignMethodPanel's render
    ("Cannot read properties of undefined (reading 'replace')"). The server
    now fills every missing category with 'Auto' before persisting, so the
    stored vd_states is always complete regardless of which client wrote it.
    """
    client, _ = app_client
    r = client.post(
        "/profiles",
        data={"name": "Partial", "kind": "design", "vd_states": json.dumps({"Gender": "male"})},
    )
    assert r.status_code == 200, r.text
    profile = client.get(f"/profiles/{r.json()['id']}").json()
    stored = json.loads(profile["vd_states"])
    assert set(stored) == {"Gender", "Age", "Pitch", "Style", "EnglishAccent", "ChineseDialect"}
    assert stored["Gender"] == "male"
    for cat in ("Age", "Pitch", "Style", "EnglishAccent", "ChineseDialect"):
        assert stored[cat] == "Auto"


# ── Migration 0005 ───────────────────────────────────────────────────────────

def _run_alembic(direction: str, db_path: str, target: str = "head"):
    from alembic import command
    from alembic.config import Config

    here = os.path.abspath(os.path.dirname(__file__))
    root = here
    while root and root != "/" and not os.path.isfile(os.path.join(root, "alembic.ini")):
        root = os.path.dirname(root)
    assert os.path.isfile(os.path.join(root, "alembic.ini")), "alembic.ini not found"
    cfg = Config(os.path.join(root, "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    if direction == "upgrade":
        command.upgrade(cfg, target)
    else:
        command.downgrade(cfg, target)


def _columns(db, table):
    with sqlite3.connect(str(db)) as conn:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _make_0003_db(db):
    """A DB as it exists after 0003 (no kind/vd_states) with one legacy row."""
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            """CREATE TABLE voice_profiles (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, ref_audio_path TEXT,
                ref_text TEXT DEFAULT '', instruct TEXT DEFAULT '',
                language TEXT DEFAULT 'Auto', locked_audio_path TEXT DEFAULT '',
                seed INTEGER DEFAULT NULL, is_locked INTEGER DEFAULT 0,
                personality TEXT DEFAULT '', description TEXT DEFAULT '',
                is_demo INTEGER DEFAULT 0, verified_own_voice INTEGER DEFAULT 0,
                consent_text TEXT DEFAULT '', consent_audio_path TEXT DEFAULT '',
                consent_recorded_at REAL DEFAULT NULL, created_at REAL
            )"""
        )
        conn.execute(
            "INSERT INTO voice_profiles (id, name, ref_audio_path, created_at) "
            "VALUES ('legacy01', 'Old Voice', 'legacy01.wav', 1.0)"
        )
        conn.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"
        )
        conn.execute(
            "INSERT INTO alembic_version VALUES ('0003_voice_profile_consent')"
        )


def test_migration_0005_adds_columns_and_backfills(tmp_path):
    db = tmp_path / "up.db"
    _make_0003_db(db)
    _run_alembic("upgrade", str(db))
    cols = _columns(db, "voice_profiles")
    assert {"kind", "vd_states"} <= cols
    with sqlite3.connect(str(db)) as conn:
        kind = conn.execute(
            "SELECT kind FROM voice_profiles WHERE id='legacy01'"
        ).fetchone()[0]
    assert kind == "clone"


def test_migration_0005_downgrade_drops_columns(tmp_path):
    db = tmp_path / "down.db"
    _make_0003_db(db)
    _run_alembic("upgrade", str(db))
    _run_alembic("downgrade", str(db), target="0003_voice_profile_consent")
    cols = _columns(db, "voice_profiles")
    assert "kind" not in cols and "vd_states" not in cols
