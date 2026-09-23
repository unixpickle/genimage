"""Qwen 2.1 image editing and reference conditioning using the official pipeline."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

from PIL import Image

from .config import Settings


def load_edit_model(settings: Settings):
    import torch
    from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline

    class CPUEncodePipeline(QwenImage21Pipeline):
        def _encode_vae_image(self, image, generator):
            return encode_reference_on_cpu(self._cpu_vae, image)

    if not torch.backends.mps.is_available():
        raise RuntimeError("Qwen Image editing requires Apple Silicon with Metal support")
    pipe = CPUEncodePipeline.from_pretrained(
        str(settings.model_dir), dtype=torch.bfloat16, local_files_only=True,
    ).to("mps")
    # Qwen 2.1's tiled decoder produces colored streaks even in a plain
    # encode/decode round trip. Keep both stages on the full-frame path.
    pipe.vae.disable_tiling()
    # Large reference images silently corrupt on MPS. Keep a separate fp32
    # encoder loaded from the original weights; the decoder stays on MPS.
    cpu_vae = AutoencoderKLQwenImage21.from_pretrained(
        str(settings.model_dir / "vae"), torch_dtype=torch.float32, local_files_only=True,
    ).eval()
    cpu_vae.disable_tiling()
    cpu_vae.decoder = None
    cpu_vae.post_quant_conv = None
    pipe._cpu_vae = cpu_vae
    return pipe


def encode_reference_on_cpu(vae, image):
    import torch

    with torch.no_grad():
        latents = vae.encode(image.to(device="cpu", dtype=torch.float32)).latent_dist.mode()
        mean = latents.new_tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = latents.new_tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        return ((latents - mean) / std).to(device=image.device, dtype=image.dtype)


def prepare_edit(job: dict[str, Any]) -> tuple[str, list[Image.Image]]:
    if job["mode"] not in {"edit", "reference"}:
        raise ValueError("Unsupported editing mode. Use Edit or References.")

    def read(path: str, mode: str = "RGB") -> Image.Image:
        with Image.open(path) as image:
            return image.convert(mode)

    source = read(job["input_path"]) if job.get("input_path") else None
    images = [source] if source is not None else []
    images.extend(read(path) for path in json.loads(job.get("reference_paths") or "[]"))
    return job["prompt"], images


def generate_edit(model, job: dict[str, Any], on_step: Callable) -> Image.Image:
    import torch

    prompt, images = prepare_edit(job)
    width, height = job["width"], job["height"]
    image = model(
        prompt=prompt,
        image=images,
        negative_prompt=job["negative_prompt"] or None,
        true_cfg_scale=job["guidance"],
        width=width,
        height=height,
        output_resolution=max(256, round(math.sqrt(job["width"] * job["height"]) / 32) * 32),
        num_inference_steps=job["steps"],
        generator=torch.Generator("cpu").manual_seed(job["seed"]),
        callback_on_step_end=on_step,
        callback_on_step_end_tensor_inputs=[],
    ).images[0]
    # Qwen natively generates RGBA. Keep alpha rather than exposing hidden RGB.
    return image
