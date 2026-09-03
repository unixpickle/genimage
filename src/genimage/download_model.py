from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

from .config import Settings


def main() -> None:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Download the configured Qwen-Image MLX checkpoint")
    parser.add_argument("--repo", default=settings.model_repo)
    parser.add_argument("--destination", default=str(settings.model_dir))
    args = parser.parse_args()
    print(f"Downloading {args.repo} to {args.destination}", flush=True)
    path = snapshot_download(
        repo_id=args.repo,
        local_dir=args.destination,
        allow_patterns=["*.json", "*.safetensors", "README.md", ".gitattributes"],
    )
    print(f"Checkpoint ready at {path}")


if __name__ == "__main__":
    main()
