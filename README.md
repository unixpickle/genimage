# GenImage

A local, persistent web UI for Qwen-Image-2512 on Apple Silicon. The web server stays responsive while a separate worker daemon owns MLX/model inference. Queued and active jobs can be cancelled or permanently deleted, and completed generations persist across restarts.

## Setup

```bash
uv sync
uv run genimage-download-model
uv run genimage
```

Open <http://127.0.0.1:8000> locally, or use the Mac's network address from another device. The server listens on `0.0.0.0:8000` by default. The default checkpoint is the 8-bit MLX conversion of Qwen-Image-2512 and is stored at:

```text
/Volumes/MLData3/genimage/models/qwen-image-2512-8bit-mlx
```

Generation state and output images live in `var/` by default. Override paths or the listen address with:

```bash
GENIMAGE_DATA_DIR=/somewhere/persistent \
GENIMAGE_MODEL_DIR=/path/to/checkpoint \
GENIMAGE_HOST=0.0.0.0 \
GENIMAGE_PORT=8000 \
uv run genimage
```

The checkpoint is loaded lazily when the first job starts. Initial loading and generation can take several minutes. Cancelling an active job signals the generation daemon and records it as cancelled; deleting it cancels and then removes its database row, uploaded source image, temporary data, and generated output.

## Development

```bash
uv run pytest
```
