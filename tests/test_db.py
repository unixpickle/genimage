from __future__ import annotations

from pathlib import Path

from genimage.config import Settings
from genimage.db import Database


def settings_for(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "model",
        model_repo="test/model",
        disable_worker=True,
        host="127.0.0.1",
        port=8000,
    )


def job_values(**overrides):
    values = {
        "prompt": "A tiny observatory",
        "negative_prompt": "blurry",
        "width": 1024,
        "height": 1024,
        "steps": 20,
        "guidance": 4.0,
        "seed": 42,
        "scheduler": "linear",
        "input_path": None,
        "image_strength": None,
        "pid_decode": 0,
        "pid_degrade_sigma": 0.0,
    }
    values.update(overrides)
    return values


def test_queue_claim_progress_and_finish_persist(tmp_path):
    db = Database(settings_for(tmp_path))
    db.initialize()
    first = db.create_job(job_values(prompt="first"))
    second = db.create_job(job_values(prompt="second"))

    claimed = db.claim_next(123)
    assert claimed["id"] == first["id"]
    assert db.update_progress(first["id"], 4, 20) is False
    output = db.settings.output_dir / f"{first['id']}.png"
    output.write_bytes(b"image")
    db.finish(first["id"], "completed", output_path=str(output))

    restarted = Database(settings_for(tmp_path))
    restarted.initialize()
    state = restarted.state()
    assert [job["id"] for job in state["queue"]] == [second["id"]]
    assert state["history"][0]["id"] == first["id"]
    assert state["history"][0]["progress_current"] == 20


def test_delete_removes_record_input_output_and_sidecar(tmp_path):
    db = Database(settings_for(tmp_path))
    db.initialize()
    input_path = db.settings.input_dir / "source.png"
    output_path = db.settings.output_dir / "result.png"
    input_path.write_bytes(b"input")
    output_path.write_bytes(b"output")
    Path(f"{output_path}.json").write_text("metadata")
    job = db.create_job(job_values(input_path=str(input_path)))
    db.finish(job["id"], "completed", output_path=str(output_path))

    found, pid = db.request_delete(job["id"])
    assert found is True and pid is None
    assert db.get_job(job["id"]) is None
    assert not input_path.exists()
    assert not output_path.exists()
    assert not Path(f"{output_path}.json").exists()


def test_active_delete_is_hidden_then_purged(tmp_path):
    db = Database(settings_for(tmp_path))
    db.initialize()
    job = db.create_job(job_values())
    db.claim_next(456)

    found, pid = db.request_delete(job["id"])
    assert found is True and pid == 456
    assert db.state()["queue"] == []
    assert db.is_cancel_requested(job["id"]) is True
    db.purge(job["id"])
    assert db.get_job(job["id"]) is None


def test_recovery_requeues_running_and_purges_deleting(tmp_path):
    db = Database(settings_for(tmp_path))
    db.initialize()
    recover = db.create_job(job_values(prompt="recover"))
    db.claim_next(111)
    deleting = db.create_job(job_values(prompt="delete"))
    db.claim_next(111)  # The first job is still running, but the single worker invariant is external.
    with db.connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'running', delete_requested = 1 WHERE id = ?", (deleting["id"],)
        )

    db.recover_incomplete()
    assert db.get_job(recover["id"])["status"] == "queued"
    assert db.get_job(deleting["id"]) is None

