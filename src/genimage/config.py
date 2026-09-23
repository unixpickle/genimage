from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

MODEL_DEFAULTS = {
    "2.1": ("Qwen/Qwen-Image-2.1", "qwen-image-2.1"),
    "2512": ("machiabeli/Qwen-Image-2512-8bit-MLX", "qwen-image-2512-8bit-mlx"),
}
MODEL_ROOT = Path("/Volumes/MLData3/genimage/models")
DEFAULT_MODEL_REPO = MODEL_DEFAULTS["2.1"][0]
DEFAULT_MODEL_DIR = MODEL_ROOT / MODEL_DEFAULTS["2.1"][1]


def checkpoint_ready(path: Path) -> bool:
    """Recognize both the official 2.1 and the original MFLUX checkpoint layouts."""
    if (path / "processor").is_dir():
        required = (
            "model_index.json", "processor/tokenizer.json", "processor/tokenizer_config.json",
            "text_encoder/config.json", "transformer/config.json", "vae/config.json",
            "scheduler/scheduler_config.json",
        )
        if not all((path / name).is_file() for name in required):
            return False
        return all(_component_ready(path / name) for name in ("text_encoder", "transformer", "vae"))
    required = (
        path / "tokenizer" / "tokenizer.json",
        path / "text_encoder" / "model.safetensors.index.json",
        path / "transformer" / "model.safetensors.index.json",
        path / "vae" / "model.safetensors.index.json",
    )
    if not all(item.is_file() for item in required):
        return False
    return all(_index_ready(index_path) for index_path in required[1:])


def _index_ready(index_path: Path) -> bool:
    try:
        index = json.loads(index_path.read_text())
        shards = set(index["weight_map"].values())
        return bool(shards) and all(
            (index_path.parent / shard).is_file() and (index_path.parent / shard).stat().st_size > 0
            for shard in shards
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def _component_ready(path: Path) -> bool:
    for stem in ("model", "diffusion_pytorch_model"):
        index = path / f"{stem}.safetensors.index.json"
        if index.is_file():
            return _index_ready(index)
        weights = path / f"{stem}.safetensors"
        if weights.is_file() and weights.stat().st_size > 0:
            return True
    return False


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    model_dir: Path
    model_repo: str
    disable_worker: bool
    host: str
    port: int
    model_variant: str = "2.1"

    @property
    def default_steps(self) -> int:
        return 40 if self.model_variant == "2.1" else 20

    @property
    def default_guidance(self) -> float:
        return 1.0 if self.model_variant == "2.1" else 4.0

    @property
    def database_path(self) -> Path:
        return self.data_dir / "genimage.sqlite3"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def input_dir(self) -> Path:
        return self.data_dir / "inputs"

    @property
    def worker_lock_path(self) -> Path:
        return self.data_dir / "worker.lock"

    @classmethod
    def from_env(cls) -> Settings:
        project_root = Path(__file__).resolve().parents[2]
        variant = os.environ.get("GENIMAGE_MODEL", "2.1")
        if variant not in MODEL_DEFAULTS:
            raise ValueError("GENIMAGE_MODEL must be 2.1 or 2512")
        repo, directory = MODEL_DEFAULTS[variant]
        return cls(
            data_dir=Path(os.environ.get("GENIMAGE_DATA_DIR", project_root / "var")).expanduser().resolve(),
            model_dir=Path(os.environ.get("GENIMAGE_MODEL_DIR", MODEL_ROOT / directory)).expanduser().resolve(),
            model_repo=os.environ.get("GENIMAGE_MODEL_REPO", repo),
            disable_worker=os.environ.get("GENIMAGE_DISABLE_WORKER", "").lower() in {"1", "true", "yes"},
            host=os.environ.get("GENIMAGE_HOST", "0.0.0.0"),
            port=int(os.environ.get("GENIMAGE_PORT", "8000")),
            model_variant=variant,
        )

    def create_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
