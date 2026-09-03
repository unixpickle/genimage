from __future__ import annotations

import asyncio
import io
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
from PIL import Image, UnidentifiedImageError

from .config import Settings, checkpoint_ready
from .db import Database, signal_worker_for_job

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
STATIC_DIR = Path(__file__).parent / "static"


def _public_job(job: dict) -> dict:
    public = {key: value for key, value in job.items() if key not in {"input_path", "output_path", "worker_pid"}}
    public["has_input"] = bool(job.get("input_path"))
    public["input_image_url"] = f"/api/jobs/{job['id']}/input" if job.get("input_path") else None
    public["image_url"] = f"/api/jobs/{job['id']}/image" if job.get("output_path") else None
    public["download_url"] = f"/api/jobs/{job['id']}/image?download=1" if job.get("output_path") else None
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
        steps: Annotated[int, Form(ge=1, le=100)] = 20,
        guidance: Annotated[float, Form(ge=0, le=20)] = 4.0,
        seed: Annotated[int | None, Form(ge=0, le=4294967295)] = None,
        scheduler: Annotated[str, Form()] = "linear",
        image_strength: Annotated[float, Form(ge=0, le=1)] = 0.75,
        pid_decode: Annotated[bool, Form()] = False,
        pid_degrade_sigma: Annotated[float, Form(ge=0, le=1)] = 0.0,
        input_image: Annotated[UploadFile | None, File()] = None,
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

        input_path = None
        if input_image and input_image.filename:
            raw = await input_image.read(MAX_UPLOAD_BYTES + 1)
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Input image is larger than 25 MB")
            try:
                with Image.open(io.BytesIO(raw)) as source:
                    source.load()
                    normalized = source.convert("RGB")
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise HTTPException(422, "The uploaded file is not a usable image") from exc
            input_path = settings.input_dir / f"{uuid.uuid4()}.png"
            normalized.save(input_path, format="PNG", optimize=True)

        try:
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
                    "input_path": str(input_path) if input_path else None,
                    "image_strength": image_strength if input_path else None,
                    "pid_decode": int(pid_decode),
                    "pid_degrade_sigma": pid_degrade_sigma,
                }
            )
        except Exception:
            if input_path:
                input_path.unlink(missing_ok=True)
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

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run("genimage.server:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
