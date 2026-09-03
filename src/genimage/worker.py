from __future__ import annotations

import fcntl
import logging
import os
import signal
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from .config import Settings, checkpoint_ready
from .db import Database

log = logging.getLogger("genimage.worker")


class GenerationCancelled(Exception):
    pass


class WorkerShutdown(BaseException):
    pass


class JobProgress:
    def __init__(self, db: Database, job_id: str, total: int):
        self.db = db
        self.job_id = job_id
        self.total = total

    def call_before_loop(self, **_: Any) -> None:
        if self.db.update_progress(self.job_id, 0, self.total, "Generating"):
            raise GenerationCancelled

    def call_in_loop(self, t: int, **_: Any) -> None:
        if self.db.update_progress(self.job_id, int(t) + 1, self.total, "Generating"):
            raise GenerationCancelled


def load_model(settings: Settings):
    if not checkpoint_ready(settings.model_dir):
        raise FileNotFoundError(
            f"Checkpoint not found at {settings.model_dir}. Run `uv run genimage-download-model`."
        )
    import mlx.core as mx
    from mflux.models.common.config import ModelConfig
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

    # Bound the reusable Metal allocation cache while retaining the model itself.
    mx.set_cache_limit(8_000_000_000)
    return QwenImage(model_path=str(settings.model_dir), model_config=ModelConfig.qwen_image())


def generate(model, db: Database, job: dict[str, Any], output_path: Path) -> None:
    callback = JobProgress(db, job["id"], job["steps"])
    model.callbacks.register(callback)
    try:
        image = model.generate_image(
            seed=job["seed"],
            prompt=job["prompt"],
            negative_prompt=job["negative_prompt"],
            width=job["width"],
            height=job["height"],
            guidance=job["guidance"],
            scheduler=job["scheduler"],
            image_path=job["input_path"],
            image_strength=job["image_strength"],
            num_inference_steps=job["steps"],
            pid_decode=bool(job["pid_decode"]),
            pid_degrade_sigma=job["pid_degrade_sigma"],
        )
        if db.is_cancel_requested(job["id"]):
            raise GenerationCancelled
        db.update_progress(job["id"], job["steps"], job["steps"], "Saving")
        image.save(path=output_path, export_json_metadata=False)
    finally:
        # Each job gets its own callback; leaving it registered leaks DB/job references.
        for collection in (
            model.callbacks.before_loop,
            model.callbacks.in_loop,
            model.callbacks.after_loop,
            model.callbacks.interrupt,
        ):
            if callback in collection:
                collection.remove(callback)


def _finish_cancelled_or_deleted(db: Database, job_id: str) -> None:
    current = db.get_job(job_id)
    if not current:
        return
    if current["delete_requested"]:
        db.purge(job_id)
    else:
        db.finish(job_id, "cancelled")


def run_worker(settings: Settings) -> int:
    settings.create_directories()
    db = Database(settings)
    db.initialize()

    lock_file = settings.worker_lock_path.open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.info("Another generation worker already owns %s", settings.worker_lock_path)
        return 0

    pid = os.getpid()
    for orphan in settings.output_dir.glob(".*.tmp.png"):
        orphan.unlink(missing_ok=True)
    db.recover_incomplete()
    db.set_worker_state(pid=pid, current_job_id=None, model_status="Not loaded")
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(2):
            try:
                db.heartbeat(pid)
            except Exception:
                log.exception("Could not update worker heartbeat")

    heartbeat_thread = threading.Thread(target=heartbeat, name="worker-heartbeat", daemon=True)
    heartbeat_thread.start()
    model = None
    model_status = "Not loaded"
    model_error = None

    try:
        while True:
            job = db.claim_next(pid)
            if job is None:
                db.set_worker_state(
                    pid=pid, current_job_id=None, model_status=model_status, model_error=model_error
                )
                time.sleep(0.75)
                continue

            job_id = job["id"]
            db.set_worker_state(pid=pid, current_job_id=job_id, model_status=model_status, model_error=model_error)
            temp_path = settings.output_dir / f".{job_id}.tmp.png"
            output_path = settings.output_dir / f"{job_id}.png"
            temp_path.unlink(missing_ok=True)
            try:
                if model is None:
                    db.update_progress(job_id, 0, job["steps"], "Loading checkpoint")
                    model_status = "Loading"
                    db.set_worker_state(pid=pid, current_job_id=job_id, model_status=model_status)
                    model = load_model(settings)
                    model_status = "Ready"
                    model_error = None
                    db.set_worker_state(pid=pid, current_job_id=job_id, model_status=model_status)
                generate(model, db, job, temp_path)
                if db.is_cancel_requested(job_id):
                    raise GenerationCancelled
                os.replace(temp_path, output_path)
                if db.is_cancel_requested(job_id):
                    output_path.unlink(missing_ok=True)
                    raise GenerationCancelled
                db.finish(job_id, "completed", output_path=str(output_path))
            except GenerationCancelled:
                temp_path.unlink(missing_ok=True)
                output_path.unlink(missing_ok=True)
                _finish_cancelled_or_deleted(db, job_id)
            except KeyboardInterrupt:
                # The web process sends SIGINT for prompt cancellation. MFLUX may turn it
                # into its own exception inside the denoising loop, or it can arrive while
                # the model is loading/decoding and reach here directly.
                temp_path.unlink(missing_ok=True)
                output_path.unlink(missing_ok=True)
                if db.is_cancel_requested(job_id):
                    _finish_cancelled_or_deleted(db, job_id)
                else:
                    db.finish(job_id, "failed", error="Generation interrupted")
                if model_status == "Loading":
                    model = None
                    model_status = "Not loaded"
            except Exception as exc:
                temp_path.unlink(missing_ok=True)
                output_path.unlink(missing_ok=True)
                if db.is_cancel_requested(job_id):
                    _finish_cancelled_or_deleted(db, job_id)
                else:
                    error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                    db.finish(job_id, "failed", error=error[-4000:])
                    log.exception("Generation %s failed", job_id)
                    if model_status == "Loading":
                        model_error = error[-1000:]
                        model_status = "Load failed"
                        model = None
            finally:
                try:
                    import mlx.core as mx

                    mx.clear_cache()
                except ImportError:
                    pass
                db.set_worker_state(
                    pid=pid, current_job_id=None, model_status=model_status, model_error=model_error
                )
    finally:
        stop_heartbeat.set()
        db.set_worker_state(pid=None, current_job_id=None, model_status="Stopped", model_error=model_error)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # The parent starts us in a new session, so Ctrl-C remains owned by the web server.
    # SIGTERM is kept distinct from per-job SIGINT cancellation so it exits the loop.
    def handle_shutdown(*_: Any) -> None:
        raise WorkerShutdown

    signal.signal(signal.SIGTERM, handle_shutdown)
    try:
        raise SystemExit(run_worker(Settings.from_env()))
    except WorkerShutdown:
        raise SystemExit(0) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
