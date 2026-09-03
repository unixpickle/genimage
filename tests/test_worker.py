from __future__ import annotations

from pathlib import Path

import pytest
from mflux.callbacks.callback_registry import CallbackRegistry

from genimage.config import Settings
from genimage.db import Database
from genimage.worker import GenerationCancelled, generate


class NeverGeneratedImage:
    def save(self, **_):
        raise AssertionError("A cancelled generation must not save an image")


class FakeModel:
    def __init__(self):
        self.callbacks = CallbackRegistry()

    def generate_image(self, **kwargs):
        context = self.callbacks.start(seed=kwargs["seed"], prompt=kwargs["prompt"], config=object())
        context.before_loop(None)
        return NeverGeneratedImage()


def test_generate_observes_cancel_and_unregisters_callback(tmp_path: Path):
    settings = Settings(tmp_path, tmp_path / "model", "test/model", True, "127.0.0.1", 8000)
    db = Database(settings)
    db.initialize()
    job = db.create_job(
        {
            "prompt": "cancel me",
            "negative_prompt": "",
            "width": 512,
            "height": 512,
            "steps": 2,
            "guidance": 4.0,
            "seed": 42,
            "scheduler": "linear",
            "input_path": None,
            "image_strength": None,
            "pid_decode": 0,
            "pid_degrade_sigma": 0.0,
        }
    )
    job = db.claim_next(123)
    db.request_cancel(job["id"])
    model = FakeModel()

    with pytest.raises(GenerationCancelled):
        generate(model, db, job, tmp_path / "never.png")

    assert model.callbacks.before_loop == []
    assert model.callbacks.in_loop == []
