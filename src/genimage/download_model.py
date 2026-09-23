from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import Settings, checkpoint_ready


def main() -> None:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Download the configured Qwen-Image MLX checkpoint")
    parser.add_argument("--repo", default=settings.model_repo)
    parser.add_argument("--destination", default=str(settings.model_dir))
    args = parser.parse_args()
    destination = Path(args.destination).expanduser().resolve()
    if str(destination).startswith("/Volumes/") and not Path(*destination.parts[:3]).is_mount():
        parser.error(f"Checkpoint volume is not mounted: {Path(*destination.parts[:3])}")
    # Keep download staging/cache data on the checkpoint's volume as well.
    os.environ.setdefault("HF_XET_CACHE", str(destination.parent.parent / ".cache" / "xet"))
    from huggingface_hub import snapshot_download

    print(f"Downloading {args.repo} to {args.destination}", flush=True)
    path = snapshot_download(
        repo_id=args.repo,
        local_dir=destination,
    )
    if not checkpoint_ready(Path(path)):
        raise RuntimeError(f"Downloaded checkpoint is incomplete or unsupported: {path}")
    print(f"Checkpoint ready at {path}")


if __name__ == "__main__":
    main()
