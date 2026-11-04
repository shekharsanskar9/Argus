"""Argus inference pipeline, ported from notebooks/helmet_violation_detection.ipynb.

`run_image` and `run_video` are generators. They yield one event (a dict) each
time a major stage starts, makes progress or finishes, so the web UI can show
the pipeline live. Stage ids match the plans in IMAGE_PLAN / VIDEO_PLAN.
"""
import base64
import csv
import os
import re
import subprocess
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# Weights are gitignored; on startup they are downloaded from HELMET_MODEL_URL /
# SEATBELT_MODEL_URL (set in the Render dashboard) into weights/.
WEIGHTS_DIR = ROOT / "weights"
HELMET_WEIGHTS = WEIGHTS_DIR / "helmet_best.pt"
SEATBELT_WEIGHTS = WEIGHTS_DIR / "seatbelt_best.pt"

# Thresholds (same values as the notebook)
DET_CONF = 0.35            # image mode: riders / helmets / plates
RIDER_CONF = 0.25          # video mode: riders / helmets
PLATE_CONF = 0.10          # video mode: plates are faint, detect aggressively
SB_CONF = 0.35
REVIEW_CONF = 0.45         # plate confidence below this -> needs human review
CONFIRM_MIN_FRAMES = 5     # N-of-M: track seen in at least N frames ...
CONFIRM_RATIO = 0.5        # ... and helmet-less in at least this fraction

# Video budget (Render runs on CPU, so cap the work per upload; tunable as env vars)
PROCESS_FPS = float(os.environ.get("ARGUS_PROCESS_FPS", 8))
MAX_SECONDS = float(os.environ.get("ARGUS_MAX_SECONDS", 30))
MAX_OCR_SAMPLES = 5
VIDEO_LONG_SIDE = 1280

ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
BRAND_WORDS = {"YAMAHA", "HONDA", "HERO", "BAJAJ", "TVS", "ROYAL"}

RED, ORANGE, MAGENTA, GREEN, YELLOW = (0, 0, 255), (0, 165, 255), (255, 0, 255), (80, 200, 60), (0, 215, 255)
CLASS_COLORS = {"rider": (255, 170, 60), "with helmet": GREEN, "without helmet": RED, "number plate": ORANGE}

IMAGE_PLAN = [
    ("load", "Load image", "Decode the upload and check its size."),
    ("detect", "Detect objects", "YOLOv8 finds riders, helmets, bare heads and number plates."),
    ("violations", "Flag helmet violations", "A rider is a violator when a 'without helmet' head sits inside the rider box."),
    ("plates", "Match & crop plates", "Pair each violator with the plate inside their box (else the nearest plate)."),
    ("ocr", "Read plates (OCR)", "Upscale 4x, build 5 pre-processed variants, run EasyOCR on each and rank candidates."),
    ("grammar", "Validate plate format", "Position-aware correction against the Indian format SS DD L NNNN + state codes."),
    ("seatbelt", "Seat-belt check", "Second YOLOv8 model flags occupants without a seat belt."),
    ("report", "Build report", "Annotated image, violations table and CSV."),
]

VIDEO_PLAN = [
    ("load", "Load video", "Read frame rate and length, pick frames to sample."),
    ("track", "Detect & track riders", "YOLOv8 + ByteTrack on each sampled frame; plate crops collected per rider."),
    ("confirm", "Confirm violations (N-of-M)", f"Keep riders seen in >= {CONFIRM_MIN_FRAMES} frames and helmet-less in >= {int(CONFIRM_RATIO * 100)}% of them."),
    ("vote", "Vote plate characters", "Confidence-weighted, character-by-character vote across every frame of a rider."),
    ("grammar", "Validate plate format", "Position-aware correction against the Indian format SS DD L NNNN + state codes."),
    ("seatbelt", "Seat-belt check", "Second YOLOv8 model flags occupants without a seat belt."),
    ("render", "Render annotated video", "Draw confirmed violators with their plates and encode H.264."),
    ("report", "Build report", "Violators table, evidence crops and CSV."),
]


# ---------------------------------------------------------------- models

def _norm(s):
    return " ".join(s.lower().replace("_", " ").replace("-", " ").split())


def find_in(names, *aliases):
    al = {_norm(a) for a in aliases}
    for i, n in names.items():
        if _norm(n) in al:
            return i
    return None


def _fetch(path, url):
    """Download weights from `url` (Google Drive share link or any direct link) if missing locally."""
    if path.exists() or not url:
        return
    m = re.search(r"drive\.google\.com/(?:file/d/|open\?id=|uc\?.*id=)([\w-]+)", url)
    if m:
        url = f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "argus"})
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if tmp.read_bytes()[:1] == b"<":
        tmp.unlink()
        raise RuntimeError(f"{url} returned a web page, not the model file. "
                           "Share it as 'Anyone with the link' and use the file's own link.")
    tmp.rename(path)


class Models:
    """Loads the YOLO models and EasyOCR once, in the background, at startup."""

    def __init__(self):
        self.ready = threading.Event()
        self.error = None
        self.helmet = self.seatbelt = self.reader = None
        self.RIDER = self.NO_HELMET = self.WITH_HELMET = self.PLATE = None
        self.SB_NO = None

    def load(self):
        try:
            _fetch(HELMET_WEIGHTS, os.environ.get("HELMET_MODEL_URL"))
            _fetch(SEATBELT_WEIGHTS, os.environ.get("SEATBELT_MODEL_URL"))
            if not HELMET_WEIGHTS.exists():
                raise RuntimeError("HELMET_MODEL_URL is not set. Add it in the Render dashboard "
                                   "(a Google Drive share link to the trained best.pt).")
            import easyocr
            import torch
            from ultralytics import YOLO

            self.helmet = YOLO(str(HELMET_WEIGHTS))
            names = self.helmet.names
            self.RIDER = find_in(names, "rider", "motorcyclist")
            self.NO_HELMET = find_in(names, "without helmet", "no helmet", "wo helmet")
            self.WITH_HELMET = find_in(names, "with helmet", "helmet")
            self.PLATE = find_in(names, "number plate", "license plate", "licence plate", "plate")
            if None in (self.RIDER, self.NO_HELMET, self.PLATE):
                raise RuntimeError(f"Unexpected helmet-model classes: {names}")

            if SEATBELT_WEIGHTS.exists():
                self.seatbelt = YOLO(str(SEATBELT_WEIGHTS))
                self.SB_NO = find_in(self.seatbelt.names, "no seatbelt", "without seatbelt", "no belt",
                                     "without belt", "noseatbelt", "not wearing seatbelt",
                                     "no seat belt", "without seat belt")

            self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
            self.helmet.predict(np.zeros((640, 640, 3), np.uint8), verbose=False)   # warm-up
        except Exception as e:
            self.error = str(e)
        finally:
            self.ready.set()

    def status(self):
        return {
            "ready": self.ready.is_set() and self.error is None,
            "loading": not self.ready.is_set(),
            "error": self.error,
            "helmet_model": self.helmet is not None,
            "seatbelt_model": self.seatbelt is not None and self.SB_NO is not None,
            "classes": dict(self.helmet.names) if self.helmet else {},
            "limits": {"max_seconds": MAX_SECONDS, "process_fps": PROCESS_FPS},
        }


MODELS = Models()


# ---------------------------------------------------------------- geometry

def center_in(outer, inner):
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def box_center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def plate_for_rider(rider, plates):
    """Plate whose centre is inside the rider box, else the nearest plate."""
    if not plates:
        return None
    inside = [p for p in plates if center_in(rider, p)]
    if inside:
        return inside[0]
    rx, ry = box_center(rider)
    return min(plates, key=lambda p: (box_center(p)[0] - rx) ** 2 + (box_center(p)[1] - ry) ** 2)


def crop_box(img, box, pad=0.1):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    return img[y1:y2, x1:x2]


def sharpness(img):
    """Variance of Laplacian: higher = sharper (less motion blur)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.Laplacian(g, cv2.CV_64F).var()


# ---------------------------------------------------------------- OCR

def clean_text(t):
    return re.sub(r"[^A-Z0-9]", "", t.upper())


def upscale(crop, max_side=640):
    """4x cubic upscale for small plates, capped so a large crop doesn't stall OCR."""
    f = min(4.0, max_side / max(crop.shape[:2]))
    return cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)


def ocr_variants(crop):
    img = upscale(crop)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "original": img,
        "gray": gray,
        "sharp": cv2.filter2D(gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])),
        "otsu": cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        "adaptive": cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 31, 11),
    }


def candidate_score(text, conf):
    L = len(text)
    s = conf * 100
    if L >= 8:
        s += 20
    elif L >= 6:
        s += 10
    elif L <= 3:
        s -= 30
    if text in BRAND_WORDS:
        s -= 80
    return s


def ocr_candidates(variants):
    out = []
    for method, proc in variants.items():
        for _, text, conf in MODELS.reader.readtext(proc, detail=1, allowlist=ALLOW):
            text = clean_text(text)
            if text:
                out.append({"method": method, "text": text, "confidence": float(conf),
                            "score": candidate_score(text, float(conf))})
    return sorted(out, key=lambda c: c["score"], reverse=True)


def read_plate_chars(crop):
    """OCR one crop -> (text, [per-char confidences]), reading left to right."""
    up = upscale(crop)
    cands = []
    for img in (up, cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)):
        res = MODELS.reader.readtext(img, detail=1, allowlist=ALLOW)
        if not res:
            continue
        text, confs = "", []
        for _, w, c in sorted(res, key=lambda r: r[0][0][0]):
            w = clean_text(w)
            text += w
            confs += [float(c)] * len(w)
        if text:
            cands.append((text, confs, float(np.mean(confs))))
    if not cands:
        return "", []
    best = max(cands, key=lambda x: x[2])
    return best[0], best[1]


def vote_plate(samples):
    """samples: list of (text, char_confs) from many frames -> consensus."""
    samples = [(t, c) for t, c in samples if t]
    if not samples:
        return "", 0.0
    L = Counter(len(t) for t, _ in samples).most_common(1)[0][0]
    kept = [(t, c) for t, c in samples if len(t) == L]
    out, pconf = [], []
    for i in range(L):
        tally = defaultdict(float)
        for t, c in kept:
            tally[t[i]] += c[i] if i < len(c) else 0.3
        ch = max(tally, key=tally.get)
        out.append(ch)
        pconf.append(tally[ch] / sum(tally.values()))
    return "".join(out), float(np.mean(pconf))


STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DL", "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL",
    "LA", "LD", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TS",
    "TR", "UP", "UK", "WB", "AN", "DN", "DD"}
L2D = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4", "T": "7"}
D2L = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A"}


def correct_indian_plate(text):
    """Position-aware fix. Returns (fixed_text, is_valid_format)."""
    t = clean_text(text)
    if not (9 <= len(t) <= 10):
        return t, False
    s = list(t)
    s[0], s[1] = D2L.get(s[0], s[0]), D2L.get(s[1], s[1])        # state letters
    s[2], s[3] = L2D.get(s[2], s[2]), L2D.get(s[3], s[3])        # RTO digits
    for i in range(len(s) - 4, len(s)):                          # last 4 = number
        s[i] = L2D.get(s[i], s[i])
    for i in range(4, len(s) - 4):                               # middle = series
        s[i] = D2L.get(s[i], s[i])
    fixed = "".join(s)
    valid = bool(re.match(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$", fixed)) and fixed[:2] in STATE_CODES
    return fixed, valid


# ---------------------------------------------------------------- drawing / encoding

def to_data_url(img, max_side=960, quality=82):
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        f = max_side / max(h, w)
        img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def enlarge(crop, height=120):
    f = height / max(crop.shape[0], 1)
    return cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)


def label(img, text, x, y, color):
    y = max(18, y)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 8, y), color, -1)
    cv2.putText(img, text, (x + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_box(img, box, color, text=None, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if text:
        label(img, text, x1, y1 - 2, color)


def h264(src, dst):
    """Re-encode to H.264 so browsers can play it. Falls back to the mp4v file."""
    try:
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(dst)], check=True, timeout=600)
        src.unlink()
    except Exception:
        src.rename(dst)


def write_csv(path, rows):
    if not rows:
        path.write_text("no violations\n")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- step bookkeeping

class Steps:
    def __init__(self):
        self.t0 = {}

    def start(self, sid, text=None):
        self.t0[sid] = time.perf_counter()
        return {"type": "step", "id": sid, "status": "running", "text": text}

    def done(self, sid, summary, status="done", **extra):
        ms = int((time.perf_counter() - self.t0.get(sid, time.perf_counter())) * 1000)
        return {"type": "step", "id": sid, "status": status, "ms": ms, "summary": summary, **extra}

    def skip(self, sid, summary):
        self.t0[sid] = time.perf_counter()
        return self.done(sid, summary, status="skipped")


def plan_event(mode):
    plan = IMAGE_PLAN if mode == "image" else VIDEO_PLAN
    return {"type": "plan", "mode": mode,
            "steps": [{"id": i, "title": t, "desc": d} for i, t, d in plan]}


def seatbelt_boxes(frame):
    if MODELS.seatbelt is None or MODELS.SB_NO is None:
        return []
    r = MODELS.seatbelt.predict(frame, conf=SB_CONF, verbose=False)[0]
    bx = r.boxes.xyxy.cpu().numpy()
    cl = r.boxes.cls.cpu().numpy().astype(int)
    return [b for b, c in zip(bx, cl) if c == MODELS.SB_NO]


# ---------------------------------------------------------------- image

def run_image(path, job_dir, job_url):
    M, S = MODELS, Steps()

    yield S.start("load")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError("Could not decode the image.")
    h, w = img.shape[:2]
    yield S.done("load", f"{w} x {h} px", images=[{"label": "Input", "src": to_data_url(img)}])

    yield S.start("detect")
    res = M.helmet.predict(img, conf=DET_CONF, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    classes = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    det_img = img.copy()
    for b, c, cf in zip(boxes, classes, confs):
        name = M.helmet.names[c]
        draw_box(det_img, b, CLASS_COLORS.get(_norm(name), YELLOW), f"{name} {cf:.2f}")
    counts = Counter(M.helmet.names[c] for c in classes)
    yield S.done("detect", f"{len(boxes)} objects detected" if len(boxes) else "Nothing detected",
                 metrics={k: counts.get(k, 0) for k in M.helmet.names.values()},
                 images=[{"label": "YOLOv8 detections", "src": to_data_url(det_img)}])

    yield S.start("violations")
    riders = [b for b, c in zip(boxes, classes) if c == M.RIDER]
    heads = [b for b, c in zip(boxes, classes) if c == M.NO_HELMET]
    violators = [r for r in riders if any(center_in(r, hd) for hd in heads)]
    viol_img = img.copy()
    for r in riders:
        bad = any(center_in(r, hd) for hd in heads)
        draw_box(viol_img, r, RED if bad else GREEN, "NO HELMET" if bad else "OK")
    yield S.done("violations", f"{len(violators)} of {len(riders)} riders without a helmet",
                 metrics={"riders": len(riders), "bare heads": len(heads), "violators": len(violators)},
                 images=[{"label": "Violation rule applied", "src": to_data_url(viol_img)}])

    plates = [b for b, c in zip(boxes, classes) if c == M.PLATE]
    found = []                                      # (rider, plate_box, crop)
    if violators:
        yield S.start("plates")
        for r in violators:
            pb = plate_for_rider(r, plates)
            crop = crop_box(img, pb) if pb is not None else None
            found.append((r, pb, crop if crop is not None and crop.size else None))
        crops = [{"label": f"Violator {i + 1}", "src": to_data_url(enlarge(c))}
                 for i, (_, _, c) in enumerate(found) if c is not None]
        n = len(crops)
        yield S.done("plates", f"{n} of {len(violators)} violators matched to a plate",
                     metrics={"plates in frame": len(plates), "matched": n}, images=crops)
    else:
        yield S.skip("plates", "No violators, nothing to match")

    reads = []                                      # per violator: dict
    if any(c is not None for _, _, c in found):
        yield S.start("ocr")
        rows, images = [], []
        for i, (r, pb, crop) in enumerate(found):
            if crop is None:
                reads.append({"raw": "", "conf": 0.0})
                continue
            variants = ocr_variants(crop)
            cands = ocr_candidates(variants)
            best = cands[0] if cands else {"text": "", "confidence": 0.0}
            reads.append({"raw": best["text"], "conf": best["confidence"]})
            if not images:
                images =[{"label": k, "src": to_data_url(v, max_side=480)} for k, v in variants.items()]
            for c in cands[:6]:
                rows.append([f"V{i + 1}", c["method"], c["text"], f"{c['confidence']:.2f}", f"{c['score']:.0f}"])
        read_n = sum(1 for x in reads if x["raw"])
        yield S.done("ocr", f"{read_n} plate(s) read, {len(rows)} top candidates shown",
                     images=images,
                     table={"columns": ["violator", "variant", "text", "conf", "score"], "rows": rows})
    else:
        reads = [{"raw": "", "conf": 0.0} for _ in found]
        yield S.skip("ocr", "No plate crops to read")

    final = []
    if any(x["raw"] for x in reads):
        yield S.start("grammar")
        rows = []
        for i, x in enumerate(reads):
            fixed, valid = correct_indian_plate(x["raw"]) if x["raw"] else ("", False)
            status = "auto-confirmed" if valid and x["conf"] >= REVIEW_CONF else "needs review"
            final.append({"plate": fixed, "valid": valid, "status": status})
            rows.append([f"V{i + 1}", x["raw"] or "-", fixed or "-", "yes" if valid else "no", status])
        n_valid = sum(f["valid"] for f in final)
        yield S.done("grammar", f"{n_valid} of {len(final)} readings match the Indian plate format",
                     table={"columns": ["violator", "raw OCR", "corrected", "valid", "status"], "rows": rows})
    else:
        final = [{"plate": "", "valid": False, "status": "needs review"} for _ in reads]
        yield S.skip("grammar", "No plate text to validate")

    sb = []
    if M.seatbelt is not None and M.SB_NO is not None:
        yield S.start("seatbelt")
        sb = seatbelt_boxes(img)
        sb_img = img.copy()
        for b in sb:
            draw_box(sb_img, b, MAGENTA, "NO SEATBELT")
        yield S.done("seatbelt", f"{len(sb)} occupant(s) without a seat belt",
                     images=[{"label": "Seat-belt model", "src": to_data_url(sb_img)}] if sb else [])
    else:
        yield S.skip("seatbelt", "Seat-belt model not loaded")

    yield S.start("report")
    out = img.copy()
    rows = []
    for i, ((r, pb, crop), x, f) in enumerate(zip(found, reads, final)):
        if pb is not None:
            draw_box(out, pb, ORANGE)
        draw_box(out, r, RED, f"NO HELMET | {f['plate'] or 'NO PLATE'}")
        rows.append({"violation": "no helmet", "id": f"V{i + 1}", "plate": f["plate"] or "NO PLATE",
                     "plate_raw": x["raw"], "ocr_confidence": round(x["conf"], 3),
                     "valid_format": f["valid"], "status": f["status"],
                     "crop": to_data_url(enlarge(crop)) if crop is not None else None})
    for i, b in enumerate(sb):
        draw_box(out, b, MAGENTA, "NO SEATBELT")
        rows.append({"violation": "no seat belt", "id": f"S{i + 1}", "plate": "-", "plate_raw": "",
                     "ocr_confidence": None, "valid_format": None, "status": "flagged", "crop": None})
    cv2.imwrite(str(job_dir / "annotated.jpg"), out)
    write_csv(job_dir / "violations.csv", [{k: v for k, v in r.items() if k != "crop"} for r in rows])
    yield S.done("report", f"{len(rows)} violation(s) in total")
    yield {"type": "result", "mode": "image", "rows": rows,
           "annotated": f"{job_url}/annotated.jpg", "csv": f"{job_url}/violations.csv"}


# ---------------------------------------------------------------- video

def run_video(path, job_dir, job_url):
    from ultralytics import YOLO

    M, S = MODELS, Steps()

    yield S.start("load")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Could not open the video.")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    skip = max(1, round(src_fps / PROCESS_FPS))
    last = min(total, int(MAX_SECONDS * src_fps)) if total else int(MAX_SECONDS * src_fps)
    n_proc = max(1, last // skip)
    f = min(1.0, VIDEO_LONG_SIDE / max(sw, sh, 1))
    OW, OH = int(sw * f) // 2 * 2, int(sh * f) // 2 * 2
    ok, first = cap.read()
    cap.release()
    if not ok:
        raise ValueError("The video has no readable frames.")
    dur = total / src_fps if total else 0
    note = f" (first {MAX_SECONDS:.0f}s analysed)" if dur > MAX_SECONDS else ""
    yield S.done("load", f"{sw}x{sh} @ {src_fps:.0f} fps, {dur:.1f}s{note}",
                 metrics={"source frames": total, "sampled frames": n_proc, "every Nth frame": skip,
                          "work size": f"{OW}x{OH}"},
                 images=[{"label": "First frame", "src": to_data_url(first)}])

    # --- detect + track: one pass, store evidence per track id
    yield S.start("track", "Starting tracker")
    tracker = YOLO(str(HELMET_WEIGHTS))             # fresh instance = fresh tracker state
    tracks = {}
    frames = []                                     # per sampled frame: what to draw later
    sb_hits = 0
    cap = cv2.VideoCapture(str(path))
    fi = done = 0
    t_start = time.perf_counter()
    while fi < last:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if fi % skip:
            continue
        frame = cv2.resize(frame, (OW, OH))
        done += 1
        res = tracker.track(frame, conf=RIDER_CONF, persist=True, verbose=False)[0]
        rec = {"riders": [], "seatbelt": []}
        if res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            ids = res.boxes.id.cpu().numpy().astype(int)
            heads = [b for b, c in zip(boxes, classes) if c == M.NO_HELMET]
            live = [(r, int(t), any(center_in(r, hd) for hd in heads))
                    for r, t, c in zip(boxes, ids, classes) if c == M.RIDER]
            plates = []
            if any(v for _, _, v in live):
                # plates are faint in video: a separate low-confidence pass, only when needed
                p = M.helmet.predict(frame, conf=PLATE_CONF, verbose=False)[0]
                pc = p.boxes.cls.cpu().numpy().astype(int)
                plates = [b for b, c in zip(p.boxes.xyxy.cpu().numpy(), pc) if c == M.PLATE]
            for r, tid, is_viol in live:
                T = tracks.setdefault(tid, dict(seen=0, nohelm=0, samples=[], best_sharp=-1, crop=None))
                T["seen"] += 1
                pb = None
                if is_viol:
                    T["nohelm"] += 1
                    pb = plate_for_rider(r, plates)
                    if pb is not None:
                        crop = crop_box(frame, pb, pad=0.2)
                        if crop.size:
                            sh_ = sharpness(crop)
                            if sh_ > T["best_sharp"]:
                                T["best_sharp"], T["crop"] = sh_, crop.copy()
                            if len(T["samples"]) < MAX_OCR_SAMPLES:
                                txt, cc = read_plate_chars(crop)
                                if txt:
                                    T["samples"].append((txt, cc, float(np.mean(cc))))
                rec["riders"].append((r, tid, is_viol, pb))
        rec["seatbelt"] = seatbelt_boxes(frame)
        sb_hits += len(rec["seatbelt"])
        frames.append(rec)

        ev = {"type": "progress", "id": "track", "pct": round(100 * done / n_proc),
              "text": f"Frame {done}/{n_proc} - {len(tracks)} riders tracked"}
        if done % 3 == 1 or done == n_proc:
            prev = frame.copy()
            for r, tid, v, pb in rec["riders"]:
                draw_box(prev, r, RED if v else GREEN, f"#{tid}")
                if pb is not None:
                    draw_box(prev, pb, ORANGE)
            ev["preview"] = to_data_url(prev, max_side=640, quality=70)
        yield ev
    cap.release()
    elapsed = time.perf_counter() - t_start
    n_samples = sum(len(T["samples"]) for T in tracks.values())
    yield S.done("track", f"{len(tracks)} riders tracked over {done} frames",
                 metrics={"frames": done, "rider tracks": len(tracks), "plate reads": n_samples,
                          "speed": f"{done / max(elapsed, 1e-6):.1f} fps"})

    # --- N-of-M confirmation
    yield S.start("confirm")
    confirmed, rows = [], []
    for tid, T in sorted(tracks.items()):
        ratio = T["nohelm"] / max(T["seen"], 1)
        ok_ = T["seen"] >= CONFIRM_MIN_FRAMES and ratio >= CONFIRM_RATIO
        if ok_:
            confirmed.append(tid)
        if T["nohelm"]:
            rows.append([f"#{tid}", T["seen"], T["nohelm"], f"{ratio:.0%}", "confirmed" if ok_ else "rejected"])
    flagged_once = sum(1 for T in tracks.values() if T["nohelm"])
    yield S.done("confirm", f"{len(confirmed)} confirmed, {flagged_once - len(confirmed)} "
                            f"single-frame alarm(s) rejected",
                 metrics={"riders flagged at least once": flagged_once, "confirmed violators": len(confirmed)},
                 table={"columns": ["rider", "frames seen", "no-helmet frames", "ratio", "decision"],
                        "rows": rows})

    # --- plate voting
    voted = {}
    if confirmed:
        yield S.start("vote")
        rows, images = [], []
        for tid in confirmed:
            T = tracks[tid]
            raw, vconf = vote_plate([(t, c) for t, c, _ in T["samples"]])
            single = max(T["samples"], key=lambda x: x[2])[0] if T["samples"] else ""
            voted[tid] = (raw, vconf)
            rows.append([f"#{tid}", len(T["samples"]), single or "-", raw or "-", f"{vconf:.2f}"])
            if T["crop"] is not None:
                images.append({"label": f"Rider #{tid} (sharpest)", "src": to_data_url(enlarge(T["crop"]))})
        yield S.done("vote", f"Consensus plate built for {sum(1 for r, _ in voted.values() if r)} "
                             f"of {len(confirmed)} violators",
                     images=images,
                     table={"columns": ["rider", "frames read", "best single frame", "voted", "vote conf"],
                            "rows": rows})
    else:
        yield S.skip("vote", "No confirmed violators")

    final = {}
    if any(r for r, _ in voted.values()):
        yield S.start("grammar")
        rows = []
        for tid, (raw, vconf) in voted.items():
            fixed, valid = correct_indian_plate(raw) if raw else ("", False)
            status = "auto-confirmed" if valid and vconf >= REVIEW_CONF else "needs review"
            final[tid] = {"plate": fixed, "valid": valid, "status": status}
            rows.append([f"#{tid}", raw or "-", fixed or "-", "yes" if valid else "no", status])
        n_valid = sum(f["valid"] for f in final.values())
        yield S.done("grammar", f"{n_valid} of {len(final)} plates match the Indian format",
                     table={"columns": ["rider", "voted", "corrected", "valid", "status"], "rows": rows})
    else:
        yield S.skip("grammar", "No plate text to validate")
    for tid in confirmed:
        final.setdefault(tid, {"plate": "", "valid": False, "status": "needs review"})

    if M.seatbelt is not None and M.SB_NO is not None:
        yield S.start("seatbelt")
        yield S.done("seatbelt", f"{sb_hits} no-seat-belt detection(s) across {done} frames",
                     metrics={"frames with a hit": sum(1 for r in frames if r["seatbelt"])})
    else:
        yield S.skip("seatbelt", "Seat-belt model not loaded")

    # --- render: second pass over the same sampled frames, now with final decisions
    yield S.start("render", "Drawing frames")
    raw_mp4, out_mp4 = job_dir / "raw.mp4", job_dir / "annotated.mp4"
    writer = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), src_fps / skip, (OW, OH))
    cap = cv2.VideoCapture(str(path))
    fi = k = 0
    conf_set = set(confirmed)
    while fi < last and k < len(frames):
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if fi % skip:
            continue
        frame = cv2.resize(frame, (OW, OH))
        rec = frames[k]
        k += 1
        for r, tid, v, pb in rec["riders"]:
            if tid in conf_set and v:
                if pb is not None:
                    draw_box(frame, pb, ORANGE)
                draw_box(frame, r, RED, f"NO HELMET | {final[tid]['plate'] or 'NO PLATE'}")
            elif v:
                draw_box(frame, r, YELLOW, f"#{tid} unconfirmed", 1)
        for b in rec["seatbelt"]:
            draw_box(frame, b, MAGENTA, "NO SEATBELT")
        writer.write(frame)
        if k % 10 == 0:
            yield {"type": "progress", "id": "render", "pct": round(100 * k / len(frames)),
                   "text": f"Frame {k}/{len(frames)}"}
    cap.release()
    writer.release()
    yield {"type": "progress", "id": "render", "pct": 100, "text": "Encoding H.264"}
    h264(raw_mp4, out_mp4)
    yield S.done("render", f"{k} frames written")

    yield S.start("report")
    rows = []
    for tid in confirmed:
        T, f_ = tracks[tid], final[tid]
        raw, vconf = voted.get(tid, ("", 0.0))
        rows.append({"violation": "no helmet", "id": f"#{tid}", "plate": f_["plate"] or "NO PLATE",
                     "plate_raw": raw, "ocr_confidence": round(vconf, 3), "valid_format": f_["valid"],
                     "status": f_["status"], "frames": T["seen"],
                     "nohelm_ratio": round(T["nohelm"] / max(T["seen"], 1), 2),
                     "crop": to_data_url(enlarge(T["crop"])) if T["crop"] is not None else None})
    write_csv(job_dir / "violations.csv", [{k: v for k, v in r.items() if k != "crop"} for r in rows])
    yield S.done("report", f"{len(rows)} confirmed violator(s)"
                           + (f", {sb_hits} seat-belt detection(s)" if sb_hits else ""))
    yield {"type": "result", "mode": "video", "rows": rows, "seatbelt_hits": sb_hits,
           "video": f"{job_url}/annotated.mp4", "csv": f"{job_url}/violations.csv"}
