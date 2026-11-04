"""Argus web app: serves the UI and streams pipeline progress as NDJSON."""
import json
import shutil
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import MODELS, plan_event, run_image, run_video

STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUT_DIR = Path("/tmp/argus_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = 200
JOB_TTL = 2 * 3600

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# One job at a time: the models are shared and CPU hosts have little headroom.
RUN_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(_):
    threading.Thread(target=MODELS.load, daemon=True).start()
    yield


app = FastAPI(title="Argus", lifespan=lifespan)


def _cleanup():
    now = time.time()
    for d in OUTPUT_DIR.iterdir():
        if d.is_dir() and now - d.stat().st_mtime > JOB_TTL:
            shutil.rmtree(d, ignore_errors=True)


def _line(ev):
    return json.dumps(ev) + "\n"


@app.get("/api/health")
def health():
    return MODELS.status()


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext in IMAGE_EXT:
        mode = "image"
    elif ext in VIDEO_EXT:
        mode = "video"
    else:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Upload an image or a video.")

    _cleanup()
    job = uuid.uuid4().hex[:12]
    job_dir = OUTPUT_DIR / job
    job_dir.mkdir()
    src = job_dir / f"input{ext}"
    size, limit = 0, MAX_UPLOAD_MB << 20
    with open(src, "wb") as out:
        while chunk := file.file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
            out.write(chunk)

    def stream():
        yield _line(plan_event(mode))
        if not MODELS.ready.is_set():
            yield _line({"type": "notice", "text": "Models are still loading, starting shortly..."})
            MODELS.ready.wait()
        if MODELS.error:
            yield _line({"type": "error", "message": MODELS.error})
            return
        if not RUN_LOCK.acquire(blocking=False):
            yield _line({"type": "notice", "text": "Another analysis is running, you are next in line..."})
            RUN_LOCK.acquire()
        try:
            yield _line({"type": "notice", "text": None})
            run = run_image if mode == "image" else run_video
            for ev in run(src, job_dir, f"/outputs/{job}"):
                yield _line(ev)
        except Exception as e:
            traceback.print_exc()
            yield _line({"type": "error", "message": str(e)})
        finally:
            RUN_LOCK.release()
            src.unlink(missing_ok=True)

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
