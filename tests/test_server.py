from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from genimage.config import Settings
from genimage.db import Database
from genimage.server import create_app


def settings_for(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "model",
        model_repo="test/model",
        disable_worker=True,
        host="127.0.0.1",
        port=8000,
    )


def test_create_list_and_permanently_delete_job(tmp_path):
    settings = settings_for(tmp_path)
    app = create_app(settings)
    image = io.BytesIO()
    Image.new("RGB", (32, 32), "orange").save(image, "PNG")

    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            data={
                "prompt": "A tangerine on blue velvet",
                "width": "1328",
                "height": "1328",
                "steps": "20",
                "guidance": "4",
                "scheduler": "linear",
                "image_strength": "0.7",
                "pid_decode": "false",
                "pid_degrade_sigma": "0",
            },
            files={"input_image": ("source.png", image.getvalue(), "image/png")},
        )
        assert response.status_code == 201, response.text
        job = response.json()
        assert job["has_input"] is True
        assert job["input_image_url"] == f"/api/jobs/{job['id']}/input"
        assert job["seed"] >= 0
        state = client.get("/api/state").json()
        assert state["queue"][0]["id"] == job["id"]
        input_files = list(settings.input_dir.iterdir())
        assert len(input_files) == 1
        input_response = client.get(job["input_image_url"])
        assert input_response.status_code == 200
        assert input_response.headers["content-type"] == "image/png"

        assert client.delete(f"/api/jobs/{job['id']}").status_code == 204
        assert client.get("/api/state").json()["queue"] == []
        assert not input_files[0].exists()
        assert client.get(job["input_image_url"]).status_code == 404


def test_validation_rejects_bad_dimensions(tmp_path):
    with TestClient(create_app(settings_for(tmp_path))) as client:
        response = client.post(
            "/api/jobs",
            data={"prompt": "test", "width": "1001", "height": "1024"},
        )
    assert response.status_code == 422
    assert "multiples of 16" in response.text


def test_delete_stale_running_job_purges_immediately(tmp_path):
    settings = settings_for(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.post("/api/jobs", data={"prompt": "stale job"})
        job = response.json()
        assert (job["width"], job["height"]) == (512, 512)
        job_id = job["id"]
        db = Database(settings)
        db.claim_next(999999)
        assert client.delete(f"/api/jobs/{job_id}").status_code == 204
        assert db.get_job(job_id) is None


def test_static_ui_is_served(tmp_path):
    with TestClient(create_app(settings_for(tmp_path))) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Qwen Image Studio" in response.text


@pytest.mark.parametrize("variant, steps, guidance", [("2.1", 40, 1.0), ("2512", 20, 4.0)])
def test_model_defaults_and_pid_validation(tmp_path, variant, steps, guidance):
    settings = replace(settings_for(tmp_path), model_variant=variant)
    with TestClient(create_app(settings)) as client:
        model = client.get("/api/state").json()["model"]
        assert model["variant"] == variant
        assert model["supports_pid"] == (variant == "2512")
        job = client.post("/api/jobs", data={"prompt": "a fox"}).json()
        assert (job["steps"], job["guidance"]) == (steps, guidance)
        for option in ({"pid_decode": "true"}, {"pid_degrade_sigma": "0.2"}):
            response = client.post("/api/jobs", data={"prompt": "a fox", **option})
            assert response.status_code == (422 if variant == "2.1" else 201)


def test_worker_receives_app_settings(tmp_path, monkeypatch):
    settings = replace(settings_for(tmp_path), disable_worker=False, model_variant="2512")
    calls = []

    class Process:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

        def poll(self):
            return 0

    monkeypatch.setattr("genimage.server.subprocess.Popen", Process)
    with TestClient(create_app(settings)):
        pass
    env = calls[0]["env"]
    assert env["GENIMAGE_MODEL"] == "2512"
    assert env["GENIMAGE_MODEL_DIR"] == str(settings.model_dir)
    assert env["GENIMAGE_DATA_DIR"] == str(settings.data_dir)
