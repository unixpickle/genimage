from __future__ import annotations

import io
import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from genimage.config import Settings
from genimage.db import SCHEMA, Database
from genimage.editing import encode_reference_on_cpu, generate_edit, prepare_edit
from genimage.server import create_app


def png(color="orange", size=(32, 32)):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, "PNG")
    return stream.getvalue()


@pytest.fixture
def settings(tmp_path):
    return Settings(tmp_path, tmp_path / "model", "test/model", True, "127.0.0.1", 8000)


def test_reference_encoding_uses_cpu_float32_and_normalizes_before_returning():
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    image = torch.zeros((1, 4, 1, 32, 32), dtype=torch.bfloat16, device=device)
    latents = torch.tensor([5.0, 14.0]).reshape(1, 2, 1, 1, 1)
    vae = SimpleNamespace(
        config=SimpleNamespace(latents_mean=[1.0, 2.0], latents_std=[2.0, 3.0]),
        encode=Mock(return_value=SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latents))),
    )
    result = encode_reference_on_cpu(vae, image)
    encoded_input = vae.encode.call_args.args[0]
    assert encoded_input.device.type == "cpu"
    assert encoded_input.dtype == torch.float32
    assert result.device == image.device
    assert result.dtype == image.dtype
    assert result.flatten().tolist() == [2.0, 4.0]


def test_original_database_migrates_without_losing_jobs(settings):
    settings.create_directories()
    with sqlite3.connect(settings.database_path) as conn:
        conn.executescript(SCHEMA)
        conn.execute(
            """INSERT INTO jobs (id, created_at, updated_at, status, prompt, width, height, steps, guidance, seed)
               VALUES ('old', 1, 1, 'queued', 'original prompt', 512, 512, 20, 4, 42)"""
        )
    db = Database(settings)
    db.initialize()
    db.initialize()
    job = db.get_job("old")
    assert job["prompt"] == "original prompt"
    assert job["mode"] == "generate"
    assert job["reference_paths"] == "[]"
    assert job["mask_path"] is None
    assert job["mask_feather"] == 0.0


def test_references_persist_and_delete(settings):
    with TestClient(create_app(settings)) as client:
        r = client.post(
            "/api/jobs",
            data={"prompt": "add a flower", "mode": "edit"},
            files=[
                ("input_image", ("source.png", png(), "image/png")),
                ("reference_images", ("ref.png", png("red"), "image/png")),
                ("reference_images", ("ref2.png", png("blue"), "image/png")),
            ],
        )
        assert r.status_code == 201, r.text
        job = r.json()
        assert len(job["reference_image_urls"]) == 2
        assert "reference_paths" not in job and "mask_path" not in job
        assert job["image_strength"] is None
        for url in [
            job["input_image_url"],
            *job["reference_image_urls"],
        ]:
            assert client.get(url).status_code == 200
        assert client.get(f"/api/jobs/{job['id']}/references/-1").status_code == 404
        assert client.get(f"/api/jobs/{job['id']}/references/2").status_code == 404
        restarted = Database(settings)
        restarted.initialize()
        saved = restarted.get_job(job["id"])
        assert saved["mode"] == "edit"
        assert len(json.loads(saved["reference_paths"])) == 2
        assert len(list(settings.input_dir.iterdir())) == 3
        assert client.delete(f"/api/jobs/{job['id']}").status_code == 204
        assert not list(settings.input_dir.iterdir())
        assert client.get(job["input_image_url"]).status_code == 404


@pytest.mark.parametrize(
    "data, files",
    [
        ({"mode": "edit"}, []),
        ({"mode": "reference"}, []),
        ({"mode": "inpaint"}, [("input_image", ("source.png", png(), "image/png"))]),
        (
            {"mode": "inpaint"},
            [
                ("input_image", ("source.png", png(), "image/png")),
                ("mask_image", ("mask.png", png("black"), "image/png")),
            ],
        ),
        (
            {"mode": "inpaint"},
            [
                ("input_image", ("source.png", png(), "image/png")),
                ("mask_image", ("mask.png", png("white", (16, 16)), "image/png")),
            ],
        ),
        ({"mode": "generate"}, [("reference_images", ("ref.png", png(), "image/png"))]),
        (
            {"mode": "reference"},
            [("reference_images", ("ref.png", png(), "image/png"))] * 11,
        ),
        (
            {"mode": "edit"},
            [
                ("input_image", ("source.png", png(), "image/png")),
                ("reference_images", ("ref.png", b"not an image", "image/png")),
            ],
        ),
        (
            {"mode": "edit", "width": 528},
            [("input_image", ("source.png", png(), "image/png"))],
        ),
    ],
)
def test_invalid_edit_does_not_leave_files_or_jobs(settings, data, files):
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/jobs", data={"prompt": "test", **data}, files=files
        )
        assert response.status_code == 422, response.text
        assert client.get("/api/state").json()["queue"] == []
        assert not list(settings.input_dir.iterdir())


def test_legacy_model_rejects_editing(settings):
    with TestClient(create_app(replace(settings, model_variant="2512"))) as client:
        r = client.post(
            "/api/jobs",
            data={"prompt": "edit it", "mode": "edit"},
            files={"input_image": ("source.png", png(), "image/png")},
        )
        assert r.status_code == 422
        assert not client.get("/api/state").json()["model"]["supports_editing"]


@pytest.mark.parametrize("mode", ["edit", "reference"])
def test_edit_pipeline_receives_references_and_preserves_alpha(tmp_path, mode):
    paths = {}
    for name, color in [("source", "red"), ("reference", "orange")]:
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (64, 32), color).save(path)
        paths[name] = str(path)
    job = {
        "mode": mode,
        "prompt": "a blue cup",
        "input_path": paths["source"] if mode == "edit" else None,
        "reference_paths": json.dumps([paths["reference"]]),
        "width": 512,
        "height": 512,
        "steps": 40,
        "guidance": 1.0,
        "negative_prompt": "",
        "seed": 42,
    }
    prompt, images = prepare_edit(job)
    assert prompt == job["prompt"]
    assert len(images) == (2 if mode == "edit" else 1)
    assert images[-1].getpixel((0, 0)) == (255, 165, 0)
    callback = Mock()
    generated = Image.new("RGBA", (512, 512), (165, 57, 209, 1))
    model = Mock(return_value=SimpleNamespace(images=[generated]))
    output = generate_edit(model, job, callback)
    kwargs = model.call_args.kwargs
    import inspect
    from diffusers import QwenImage21Pipeline

    inspect.signature(QwenImage21Pipeline.__call__).bind(model, **kwargs)
    assert kwargs["callback_on_step_end"] is callback
    assert kwargs["generator"].initial_seed() == 42
    assert kwargs["width"] == kwargs["height"] == 512
    assert len(kwargs["image"]) == len(images)
    assert output is generated
    assert output.mode == "RGBA"
    assert output.getpixel((0, 0)) == (165, 57, 209, 1)


def test_removed_inpainting_is_rejected_without_saving_uploads(settings):
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/jobs",
            data={"prompt": "a tile floor", "mode": "inpaint"},
            files={
                "input_image": ("source.png", png(), "image/png"),
                "mask_image": ("mask.png", png("white"), "image/png"),
            },
        )
        assert response.status_code == 422
        assert "no longer supported" in response.json()["detail"]
        assert not client.get("/api/state").json()["queue"]
        assert not list(settings.input_dir.iterdir())


def test_edit_rejects_legacy_mask_upload_instead_of_ignoring_it(settings):
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/jobs",
            data={"prompt": "a tile floor", "mode": "edit"},
            files={
                "input_image": ("source.png", png(), "image/png"),
                "mask_image": ("mask.png", png("white"), "image/png"),
            },
        )
        assert response.status_code == 422
        assert "Mask uploads" in response.json()["detail"]
        assert not list(settings.input_dir.iterdir())


def test_legacy_inpaint_history_remains_viewable_and_deletable(settings):
    with TestClient(create_app(settings)) as client:
        db = Database(settings)
        source_path = settings.input_dir / "source.png"
        mask_path = settings.input_dir / "mask.png"
        source_path.write_bytes(png())
        mask_path.write_bytes(png("white"))
        job = db.create_job({
            "mode": "inpaint", "prompt": "old edit", "width": 512, "height": 512,
            "steps": 40, "guidance": 1.0, "seed": 42,
            "input_path": str(source_path), "mask_path": str(mask_path),
        })
        db.finish(job["id"], "completed")
        saved = client.get("/api/state").json()["history"][0]
        assert saved["mode"] == "inpaint"
        assert client.get(saved["input_image_url"]).status_code == 200
        assert client.get(saved["mask_image_url"]).status_code == 200
        assert client.delete(f"/api/jobs/{job['id']}").status_code == 204
        assert not source_path.exists() and not mask_path.exists()
