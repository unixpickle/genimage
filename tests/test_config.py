import json

import pytest

from genimage.config import MODEL_ROOT, Settings, checkpoint_ready


@pytest.mark.parametrize(
    "variant, directory, steps, guidance",
    [
        ("2.1", "qwen-image-2.1", 40, 1.0),
        ("2512", "qwen-image-2512-8bit-mlx", 20, 4.0),
    ],
)
def test_model_defaults_share_volume(monkeypatch, variant, directory, steps, guidance):
    monkeypatch.setenv("GENIMAGE_MODEL", variant)
    monkeypatch.delenv("GENIMAGE_MODEL_DIR", raising=False)
    monkeypatch.delenv("GENIMAGE_MODEL_REPO", raising=False)
    settings = Settings.from_env()
    assert settings.model_dir == MODEL_ROOT / directory
    assert settings.default_steps == steps
    assert settings.default_guidance == guidance


def test_unknown_model_is_rejected(monkeypatch):
    monkeypatch.setenv("GENIMAGE_MODEL", "unknown")
    with pytest.raises(ValueError, match="GENIMAGE_MODEL"):
        Settings.from_env()


@pytest.mark.parametrize("variant", ["2.1", "2512"])
def test_checkpoint_requires_all_shards(tmp_path, variant):
    def write(name, content=b"weights"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    if variant == "2.1":
        for name in (
            "model_index.json",
            "processor/tokenizer.json",
            "processor/tokenizer_config.json",
            "text_encoder/config.json",
            "transformer/config.json",
            "vae/config.json",
            "scheduler/scheduler_config.json",
        ):
            write(name, b"{}")
        write("vae/diffusion_pytorch_model.safetensors")
        indices = [
            "text_encoder/model.safetensors.index.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
        ]
    else:
        write("tokenizer/tokenizer.json", b"{}")
        indices = [
            f"{component}/model.safetensors.index.json"
            for component in ("text_encoder", "transformer", "vae")
        ]
    for index in indices:
        write(
            index,
            json.dumps(
                {"weight_map": {"a": "shard-1.safetensors", "b": "shard-2.safetensors"}}
            ).encode(),
        )
        component = index.split("/")[0]
        write(f"{component}/shard-1.safetensors")
        write(f"{component}/shard-2.safetensors")
    assert checkpoint_ready(tmp_path)
    shard = tmp_path / "transformer/shard-2.safetensors"
    shard.unlink()
    assert not checkpoint_ready(tmp_path)
    shard.write_bytes(b"")
    assert not checkpoint_ready(tmp_path)
    shard.write_bytes(b"weights")
    (tmp_path / indices[0]).write_text('{"weight_map": null}')
    assert not checkpoint_ready(tmp_path)
