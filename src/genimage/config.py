from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_REPO = "machiabeli/Qwen-Image-2512-8bit-MLX"
DEFAULT_MODEL_DIR = Path("/Volumes/MLData3/genimage/models/qwen-image-2512-8bit-mlx")


def checkpoint_ready(path: Path) -> bool:
    """Return true only when every shard named by each component index exists."""
    required = (
        path / "tokenizer" / "tokenizer.json",
        path / "text_encoder" / "model.safetensors.index.json",
        path / "transformer" / "model.safetensors.index.json",
        path / "vae" / "model.safetensors.index.json",
    )
    if not all(item.is_file() for item in required):
        return False
    try:
        for index_path in required[1:]:
            index = json.loads(index_path.read_text())
            shards = set(index["weight_map"].values())
            if not shards or not all((index_path.parent / shard).is_file() for shard in shards):
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    model_dir: Path
    model_repo: str
    disable_worker: bool
    host: str
    port: int

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
        return cls(
            data_dir=Path(os.environ.get("GENIMAGE_DATA_DIR", project_root / "var")).expanduser().resolve(),
            model_dir=Path(os.environ.get("GENIMAGE_MODEL_DIR", DEFAULT_MODEL_DIR)).expanduser().resolve(),
            model_repo=os.environ.get("GENIMAGE_MODEL_REPO", DEFAULT_MODEL_REPO),
            disable_worker=os.environ.get("GENIMAGE_DISABLE_WORKER", "").lower() in {"1", "true", "yes"},
            host=os.environ.get("GENIMAGE_HOST", "0.0.0.0"),
            port=int(os.environ.get("GENIMAGE_PORT", "8000")),
        )

    def create_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
