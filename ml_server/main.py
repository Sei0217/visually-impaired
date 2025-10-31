# ml_server/main.py — faster CPU settings
import os, io, base64, traceback
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# -------- threads: make it 2 by default (faster on Render) --------
THREADS = int(os.getenv("NUM_THREADS", "2"))
os.environ["OMP_NUM_THREADS"] = str(THREADS)
os.environ["MKL_NUM_THREADS"] = str(THREADS)
try:
    import torch
    torch.set_num_threads(THREADS)
except Exception:
    torch = None
# -------------------------------------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "best.pt"))
print("MODEL_PATH:", MODEL_PATH, "exists:", os.path.exists(MODEL_PATH))
model = YOLO(MODEL_PATH)

# Fuse Conv+BN for small CPU speedup
try:
    model.fuse()
    print("[init] model fused")
except Exception as e:
    print("[init] fuse skipped:", e)

KEEP = {"person", "door", "stairs"}  # limit classes (less NMS work)

@app.on_event("startup")
def _warmup():
    try:
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        if torch is not None:
            with torch.no_grad():
                _ = model(dummy, imgsz=256, conf=0.25, verbose=False, max_det=20)[0]
        else:
            _ = model(dummy, imgsz=256, conf=0.25, verbose=False, max_det=20)[0]
        print("[startup] warmup done")
    except Exception as e:
        print("[startup] warmup failed:", e)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/detect")
async def detect(
    image: UploadFile = File(...),
    conf: float = 0.25,
    imgsz: int = 256,       # smaller default
    max_det: int = 20       # limit number of boxes for speed
):
    try:
        # clamp params
        try: imgsz = int(imgsz)
        except: imgsz = 256
        imgsz = max(192, min(imgsz, 384))   # keep in a tight fast range

        try: conf = float(conf)
        except: conf = 0.25
        conf = max(0.05, min(conf, 0.9))

        try: max_det = int(max_det)
        except: max_det = 20
        max_det = max(10, min(max_det, 50))

        # read->decode
        raw = await image.read()
        im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            return {"success": False, "error": "decode_failed"}

        # restrict classes by index (faster NMS)
        names = model.names
        keep_idx = [i for i, n in (names.items() if isinstance(names, dict) else enumerate(names)) if n in KEEP]

        # inference
        if torch is not None:
            with torch.inference_mode():
                r = model(im, imgsz=imgsz, conf=conf, verbose=False, max_det=max_det, classes=keep_idx)[0]
        else:
            r = model(im, imgsz=imgsz, conf=conf, verbose=False, max_det=max_det, classes=keep_idx)[0]

        # build dets
        xyxy = r.boxes.xyxy.tolist() if r.boxes is not None else []
        cls  = r.boxes.cls.tolist()  if r.boxes is not None else []
        cf   = r.boxes.conf.tolist() if r.boxes is not None else []
        names = r.names if hasattr(r, "names") else model.names

        dets, top_label, top_conf = [], "No detection", 0.0
        for bb, c, p in zip(xyxy, cls, cf):
            lbl = names[int(c)]
            p = float(p)
            dets.append({"bbox": bb, "label": lbl, "confidence": p, "conf": p})
        if dets:
            top_idx = max(range(len(dets)), key=lambda i: dets[i]["confidence"])
            top_label = dets[top_idx]["label"]; top_conf = dets[top_idx]["confidence"]

        # annotated (fast)
        annotated = r.plot()  # BGR with good built-in labels
        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            return {"success": False, "error": "encode_failed"}
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        data_url = "data:image/jpeg;base64," + b64

        return {
            "success": True,
            "annotated_image_url": data_url,
            "image_base64": data_url,
            "detections": dets,
            "object_type": top_label,
            "confidence_score": round(top_conf * 100, 2),
        }

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}

# legacy alias
@app.post("/detect")
async def detect_alias(image: UploadFile = File(...), conf: float = 0.25, imgsz: int = 256, max_det: int = 20):
    return await detect(image=image, conf=conf, imgsz=imgsz, max_det=max_det)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
