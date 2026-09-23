from __future__ import annotations

import asyncio
import io
import json
import os
import secrets
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Settings, checkpoint_ready
from .db import Database, signal_worker_for_job

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
STATIC_DIR = Path(__file__).parent / "static"


def _public_job(job: dict) -> dict:
    public = {key: value for key, value in job.items()
              if key not in {"input_path", "output_path", "worker_pid", "reference_paths", "mask_path"}}
    public["has_input"] = bool(job.get("input_path"))
    public["input_image_url"] = f"/api/jobs/{job['id']}/input" if job.get("input_path") else None
    public["image_url"] = f"/api/jobs/{job['id']}/image" if job.get("output_path") else None
    public["download_url"] = f"/api/jobs/{job['id']}/image?download=1" if job.get("output_path") else None
    references = json.loads(job.get("reference_paths") or "[]")
    public["reference_image_urls"] = [f"/api/jobs/{job['id']}/references/{i}" for i in range(len(references))]
    public["mask_image_url"] = f"/api/jobs/{job['id']}/mask" if job.get("mask_path") else None
    return public


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    db = Database(settings)
    worker_process: subprocess.Popen | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal worker_process
        db.initialize()
        if not settings.disable_worker:
            worker_process = subprocess.Popen(  # noqa: ASYNC220 - startup is intentionally synchronous
                [sys.executable, "-m", "genimage.worker"],
                start_new_session=True,
                env={
                    **os.environ,
                    "GENIMAGE_DATA_DIR": str(settings.data_dir),
                    "GENIMAGE_MODEL_DIR": str(settings.model_dir),
                    "GENIMAGE_MODEL_REPO": settings.model_repo,
                    "GENIMAGE_MODEL": settings.model_variant,
                },
            )
        yield
        if worker_process and worker_process.poll() is None:
            worker_process.terminate()
            try:
                await asyncio.to_thread(worker_process.wait, 5)
            except subprocess.TimeoutExpired:
                worker_process.kill()
                await asyncio.to_thread(worker_process.wait)

    app = FastAPI(title="GenImage", lifespan=lifespan)

    @app.get("/api/state")
    def get_state():
        state = db.state()
        state["queue"] = [_public_job(job) for job in state["queue"]]
        state["history"] = [_public_job(job) for job in state["history"]]
        state["model"] = {
            "repo": settings.model_repo,
            "path": str(settings.model_dir),
            "downloaded": checkpoint_ready(settings.model_dir),
            "variant": settings.model_variant,
            "default_steps": settings.default_steps,
            "default_guidance": settings.default_guidance,
            "supports_pid": settings.model_variant == "2512",
            "supports_editing": settings.model_variant == "2.1",
        }
        heartbeat = state["worker"].get("heartbeat")
        state["worker"]["alive"] = bool(
            state["worker"].get("pid") and heartbeat and time.time() - heartbeat < 8
        )
        return state

    @app.post("/api/jobs", status_code=201)
    async def create_job(
        prompt: Annotated[str, Form(min_length=1, max_length=8000)],
        negative_prompt: Annotated[str, Form(max_length=8000)] = "",
        width: Annotated[int, Form(ge=256, le=2048)] = 512,
        height: Annotated[int, Form(ge=256, le=2048)] = 512,
        steps: Annotated[int, Form(ge=1, le=100)] = settings.default_steps,
        guidance: Annotated[float, Form(ge=0, le=20)] = settings.default_guidance,
        seed: Annotated[int | None, Form(ge=0, le=4294967295)] = None,
        scheduler: Annotated[str, Form()] = "linear",
        image_strength: Annotated[float, Form(ge=0, le=1)] = 0.75,
        pid_decode: Annotated[bool, Form()] = False,
        pid_degrade_sigma: Annotated[float, Form(ge=0, le=1)] = 0.0,
        input_image: Annotated[UploadFile | None, File()] = None,
        mode: Annotated[str, Form()] = "generate",
        reference_images: Annotated[list[UploadFile] | None, File()] = None,
        # Reject uploads from older clients rather than silently ignoring a mask.
        mask_image: Annotated[UploadFile | None, File()] = None,
    ):
        prompt = prompt.strip()
        if not prompt:
            raise HTTPException(422, "Prompt cannot be blank")
        if width % 16 or height % 16:
            raise HTTPException(422, "Width and height must be multiples of 16")
        if width * height > 2048 * 2048:
            raise HTTPException(422, "The requested image is too large")
        if scheduler != "linear":
            raise HTTPException(422, "Qwen-Image currently supports the linear scheduler")
        if settings.model_variant == "2.1" and (pid_decode or pid_degrade_sigma):
            raise HTTPException(422, "PiD decoding is not supported by Qwen Image 2.1")
        if mode == "inpaint":
            raise HTTPException(422, "Inpainting is no longer supported. Use Edit mode.")
        if mode not in {"generate", "img2img", "edit", "reference"}:
            raise HTTPException(422, "Unknown generation mode")
        editing = mode in {"edit", "reference"}
        if editing and settings.model_variant != "2.1":
            raise HTTPException(422, "Editing and references require Qwen Image 2.1")
        if editing and (width % 32 or height % 32):
            raise HTTPException(422, "Editing dimensions must be multiples of 32")
        has_input = bool(input_image and input_image.filename)
        references = [image for image in (reference_images or []) if image.filename]
        if mask_image and mask_image.filename:
            raise HTTPException(422, "Mask uploads are no longer supported. Use Edit mode.")
        if mode in {"edit", "img2img"} and not has_input:
            raise HTTPException(422, "Choose a source image for this mode")
        if mode == "reference" and not references:
            raise HTTPException(422, "Add at least one reference image")
        if mode == "reference" and has_input:
            raise HTTPException(422, "Upload images as references in Reference mode")
        if references and not editing:
            raise HTTPException(422, "Choose Edit or References to use reference images")
        if len(references) + int(has_input) > 10:
            raise HTTPException(422, "Use at most 10 images total, including the source")

        async def read_upload(upload: UploadFile) -> Image.Image:
            raw = await upload.read(MAX_UPLOAD_BYTES + 1)
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Input image is larger than 25 MB")
            try:
                with Image.open(io.BytesIO(raw)) as source:
                    if source.width * source.height > 16_000_000:
                        raise HTTPException(422, "Input images must be at most 16 megapixels")
                    source.load()
                    return ImageOps.exif_transpose(source).convert("RGB")
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise HTTPException(422, "The uploaded file is not a usable image") from exc

        saved_paths = []
        def save_upload(image: Image.Image) -> str:
            path = settings.input_dir / f"{uuid.uuid4()}.png"
            saved_paths.append(path)
            image.save(path, format="PNG")
            return str(path)

        input_path = None
        reference_paths = []
        try:
            source = await read_upload(input_image) if has_input else None
            if source is not None:
                input_path = save_upload(source)
            for reference in references:
                reference_paths.append(save_upload(await read_upload(reference)))
            job = db.create_job(
                {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt.strip(),
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "guidance": guidance,
                    "seed": seed if seed is not None else secrets.randbelow(2**32),
                    "scheduler": scheduler,
                    "input_path": input_path,
                    "image_strength": image_strength if input_path and not editing else None,
                    "mode": "img2img" if mode == "generate" and has_input else mode,
                    "reference_paths": json.dumps(reference_paths),
                    "pid_decode": int(pid_decode),
                    "pid_degrade_sigma": pid_degrade_sigma,
                }
            )
        except Exception:
            for path in saved_paths:
                path.unlink(missing_ok=True)
            raise
        return _public_job(job)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job, active = db.request_cancel(job_id)
        if not job:
            raise HTTPException(404, "Generation not found")
        if active and not signal_worker_for_job(db, job_id):
            current = db.get_job(job_id)
            if current and current["status"] == "running":
                db.finish(job_id, "cancelled")
                job = db.get_job(job_id)
        return _public_job(job)

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str):
        found, worker_pid = db.request_delete(job_id)
        if not found:
            raise HTTPException(404, "Generation not found")
        if worker_pid and not signal_worker_for_job(db, job_id):
            db.purge(job_id)

    @app.get("/api/jobs/{job_id}/image")
    def get_image(job_id: str, download: bool = False):
        job = db.get_job(job_id)
        if not job or job["delete_requested"] or not job["output_path"]:
            raise HTTPException(404, "Image not found")
        path = Path(job["output_path"])
        if not path.is_file():
            raise HTTPException(404, "Image file not found")
        disposition = "attachment" if download else "inline"
        return FileResponse(path, media_type="image/png", filename=f"qwen-{job_id}.png", content_disposition_type=disposition)

    @app.get("/api/jobs/{job_id}/input")
    def get_input_image(job_id: str):
        job = db.get_job(job_id)
        if not job or job["delete_requested"] or not job["input_path"]:
            raise HTTPException(404, "Initial image not found")
        path = Path(job["input_path"])
        if not path.is_file():
            raise HTTPException(404, "Initial image file not found")
        return FileResponse(path, media_type="image/png", content_disposition_type="inline")

    @app.get("/api/jobs/{job_id}/references/{index}")
    def get_reference(job_id: str, index: int):
        job = db.get_job(job_id)
        paths = json.loads(job.get("reference_paths") or "[]") if job else []
        if not job or job["delete_requested"] or not 0 <= index < len(paths):
            raise HTTPException(404, "Reference image not found")
        if not Path(paths[index]).is_file():
            raise HTTPException(404, "Reference image file not found")
        return FileResponse(paths[index], media_type="image/png")

    @app.get("/api/jobs/{job_id}/mask")
    def get_mask(job_id: str):
        # Historical jobs retain their original attachments for inspection.
        job = db.get_job(job_id)
        if not job or job["delete_requested"] or not job.get("mask_path"):
            raise HTTPException(404, "Mask not found")
        if not Path(job["mask_path"]).is_file():
            raise HTTPException(404, "Mask file not found")
        return FileResponse(job["mask_path"], media_type="image/png")

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run("genimage.server:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
