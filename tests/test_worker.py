from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

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


@pytest.mark.parametrize("variant", ["2.1", "2512"])
def test_generate_dispatch_and_callback_cleanup(tmp_path, variant):
    settings = Settings(tmp_path, tmp_path / "model", "test/model", True, "127.0.0.1", 8000,
                        model_variant=variant)
    db = Database(settings)
    db.initialize()
    job = db.create_job({
        "prompt": "a fox", "negative_prompt": "blur", "width": 512, "height": 512,
        "steps": 40, "guidance": 1.0, "seed": 42, "scheduler": "linear", "input_path": None,
        "image_strength": None, "pid_decode": 0, "pid_degrade_sigma": 0.0,
    })
    model = FakeModel()
    model.generate_image = Mock(return_value=Mock())
    output = tmp_path / "output.png"
    generate(model, db, job, output)
    kwargs = model.generate_image.call_args.kwargs
    # Binding to the real API catches unsupported arguments without loading weights.
    import inspect
    if variant == "2.1":
        from mflux.models.qwen21.variants.txt2img.qwen_image_21 import (
            QwenImage21 as Model,
        )
        assert "pid_decode" not in kwargs
        assert "pid_degrade_sigma" not in kwargs
    else:
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage as Model
        assert kwargs["pid_decode"] is False
    inspect.signature(Model.generate_image).bind(model, **kwargs)
    model.generate_image.return_value.save.assert_called_once_with(path=output, export_json_metadata=False)
    assert not model.callbacks.before_loop
    assert not model.callbacks.in_loop
    model.generate_image.side_effect = RuntimeError("inference failed")
    with pytest.raises(RuntimeError, match="inference failed"):
        generate(model, db, job, output)
    assert not model.callbacks.before_loop
    assert not model.callbacks.in_loop


def test_legacy_inpaint_job_cannot_fall_back_to_text_generation(tmp_path):
    model = Mock()
    with pytest.raises(ValueError, match="Inpainting is no longer supported"):
        generate(model, Mock(), {"mode": "inpaint"}, tmp_path / "never.png")
    assert not model.mock_calls
    assert not (tmp_path / "never.png").exists()
