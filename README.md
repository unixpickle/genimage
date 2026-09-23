# GenImage

A local, persistent web UI for Qwen-Image-2.1 and Qwen-Image-2512 on Apple Silicon. The web server stays responsive while a separate worker daemon owns MLX/model inference. Queued and active jobs can be cancelled or permanently deleted, and completed generations persist across restarts.

## Setup

```bash
uv sync
uv run genimage-download-model
./run.sh
```

Open <http://127.0.0.1:8000> locally, or use the Mac's network address from another device. The server listens on `0.0.0.0:8000` by default. The default checkpoint is the official [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) in bf16 (about 33 GB), stored on the same volume as the original checkpoint:

```text
/Volumes/MLData3/genimage/models/qwen-image-2.1
```

Mount `MLData3` before downloading. Download staging and the Xet cache also use that volume. The downloader resumes interrupted downloads and checks for missing weight shards before reporting success.

Qwen 2.1 uses 40 steps and guidance 1.0 by default. Use a 64 GB Mac for the bf16 model. The UI starts at 512 × 512; larger canvases require more memory and time. This project pins MFLUX to commit `8c00dab2505a96019df9d30bc9c223bf20d733c4`, which adds its [Qwen 2.1 implementation](https://github.com/mflux-community/mflux/blob/8c00dab2505a96019df9d30bc9c223bf20d733c4/src/mflux/models/qwen21/README.md).

Choose a mode in the composer:

- **Text to image:** generate from a prompt.
- **Edit an image:** upload a source, describe a change, and optionally add references.
- **Generate from references:** combine up to 10 images; refer to them as “image 1”, “image 2”, etc. in the prompt.
- **Image variation:** use an initial image and a strength value to guide a new image.

Completed images have **Edit** and **Reference** buttons that set the canvas from the actual image dimensions, rounded to multiples of 32 and fitted within the existing 256–2048 pixel limits when necessary. **Reuse settings**, available on cards and in Details, restores the original prompt, negative prompt, mode, resolution, sampling settings, source and references. It clears the seed and enables randomization so resubmitting produces a new sample. Sources, references, and settings persist with each job and are removed when the job is deleted. Each job accepts at most 10 conditioning images including the source (9 extra references for editing). Uploads are limited to 25 MB and 16 megapixels each. Editing dimensions must be multiples of 32; presets adjust automatically.

Editing uses the official Diffusers `QwenImage21Pipeline` on Apple Metal, pinned to commit `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`. It reads the same local checkpoint as MFLUX. The worker unloads one runtime before loading the other, so switching between generation and editing may take longer. Editing uses native prompt-guided image conditioning. The app does not offer an inpainting mode; the pipeline has no dedicated inpainting input or sampling path. Historical inpainting jobs remain viewable and deletable, but cannot be resubmitted.

Reference images are VAE-encoded on the CPU in float32 using the original encoder weights. Metal encoding silently corrupts references at larger resolutions: the 1120 × 1472 round trip reproduces a desaturated, embossed image even without diffusion. CPU encoding fixes this; diffusion and VAE decoding remain on Metal. A separate CPU encoder avoids transferring the decoder for every reference. Both VAE stages process full frames with tiling disabled because the default tiles also introduce streaks and seams.

Guidance above 1 only takes effect with a nonempty negative prompt. Editing outputs preserve the model’s RGBA transparency. PiD decoding is unsupported for 2.1; the UI hides its controls and the API rejects those options. Edit and References modes require the official 2.1 checkpoint and are unavailable for 2512.

The original 8-bit checkpoint remains at `/Volumes/MLData3/genimage/models/qwen-image-2512-8bit-mlx`. To use it with its original defaults (20 steps, guidance 4):

```bash
GENIMAGE_MODEL=2512 uv run genimage
```

Use the same `GENIMAGE_MODEL=2512` prefix with `genimage-download-model` to download the original model. A server uses one model for its entire queue; stop it and finish or clear pending jobs before switching models.

Generation state and output images live in `var/` by default. Override paths or the listen address with:

```bash
GENIMAGE_DATA_DIR=/somewhere/persistent \
GENIMAGE_MODEL=2.1 \
GENIMAGE_MODEL_DIR=/path/to/checkpoint \
GENIMAGE_HOST=0.0.0.0 \
GENIMAGE_PORT=8000 \
uv run genimage
```

The checkpoint is loaded lazily when the first job starts. Initial loading and generation can take several minutes. Cancelling an active job signals the generation daemon and records it as cancelled; deleting it cancels and then removes its database row, uploaded source image, temporary data, and generated output.

## Development

```bash
uv run pytest
node --check src/genimage/static/app.js
node --check src/genimage/static/editor.js
```

Validated on a 64 GB M4 Pro with actual queued generations: 1024 × 1024 text-to-image at 40 steps, single-image editing, and two-reference composition. Unit/API tests cover checkpoint completeness, job migration/persistence, invalid uploads, cleanup, and backend arguments. Browser checks exercise references and mobile layout.
