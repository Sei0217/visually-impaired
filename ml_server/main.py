# ml_server/main.py — unified (Render-ready, r.plot() like localhost)
import os, io, base64, traceback
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from fastapi import Query
from fastapi.responses import StreamingResponse
from io import BytesIO
from gtts import gTTS
from fastapi.middleware.cors import CORSMiddleware



# -------- perf limits for small instances --------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
except Exception:
    torch = None
# -------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://visually-impaired-nine.vercel.app",
        "http://localhost:3000",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app$",
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)

@app.get("/tts")
def tts(text: str = Query(..., max_length=400),
        lang: str = Query("en")):
    # map UI lang → gTTS lang
    lang_map = {"tl": "tl", "ceb": "ceb", "en": "en", "en-PH": "en", "fil": "tl"}
    g_lang = lang_map.get(lang, "en")

    # generate mp3 in-memory
    mp3 = BytesIO()
    tts = gTTS(text=text, lang=g_lang, slow=False)
    tts.write_to_fp(mp3)
    mp3.seek(0)
    return StreamingResponse(mp3, media_type="audio/mpeg")

# Model path (relative-safe)
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "best.pt"))
print("MODEL_PATH:", MODEL_PATH, "exists:", os.path.exists(MODEL_PATH))
model = YOLO(MODEL_PATH)

KEEP = {"person", "door", "stairs"}  # keep only these labels

@app.on_event("startup")
def _warmup():
    try:
        # Small black image for first-time compile/warmup
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        if torch is not None:
            with torch.no_grad():
                _ = model(dummy, imgsz=320, conf=0.25, verbose=False)[0]
        else:
            _ = model(dummy, imgsz=320, conf=0.25, verbose=False)[0]
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
    imgsz: int = 320
):
    try:
        # sanitize inputs
        try:
            imgsz = int(imgsz)
        except Exception:
            imgsz = 320
        imgsz = max(160, min(imgsz, 640))

        try:
            conf = float(conf)
        except Exception:
            conf = 0.25
        conf = max(0.05, min(conf, 0.9))

        # decode image
        raw = await image.read()
        np_img = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            return {"success": False, "error": "decode_failed"}

        # inference (Ultralytics __call__)
        if torch is not None:
            with torch.no_grad():
                r = model(img, imgsz=imgsz, conf=conf, verbose=False)[0]
        else:
            r = model(img, imgsz=imgsz, conf=conf, verbose=False)[0]

        # build detections (filter to KEEP)
        names = r.names if hasattr(r, "names") else model.names
        dets, top_label, top_conf = [], "No detection", 0.0
        if r.boxes is not None and len(r.boxes) > 0:
            # get xyxy/cls/conf lists
            xyxy = r.boxes.xyxy.tolist()
            cls  = r.boxes.cls.tolist()
            confs = r.boxes.conf.tolist()
            for bb, c, p in zip(xyxy, cls, confs):
                label = names[int(c)]
                if label in KEEP:
                    p = float(p)
                    dets.append({
                        "bbox": bb,
                        "label": label,
                        "confidence": p,  # for new UI
                        "conf": p         # for old UI
                    })
            if dets:
                top_idx = max(range(len(dets)), key=lambda i: dets[i]["confidence"])
                top_label = dets[top_idx]["label"]
                top_conf  = dets[top_idx]["confidence"]

# --- Faster annotate via pure OpenCV ---
        annotated = img.copy()
        if dets:
            for d in dets:
                x1, y1, x2, y2 = map(int, d["bbox"])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (36, 255, 12), 2)
                label = f'{d["label"]} {int(d["confidence"]*100)}%'
                # background for text
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 6, y1), (36, 255, 12), -1)
                cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        # mas mababang JPEG quality para mas maliit ang payload pabalik
        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])

        if not ok:
            return {"success": False, "error": "encode_failed"}
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        data_url = "data:image/jpeg;base64," + b64

        return {
            "success": True,
            "annotated_image_url": data_url,   # match localhost
            "image_base64": data_url,          # also provide for compatibility
            "detections": dets,
            "object_type": top_label,
            "confidence_score": round(top_conf * 100, 2),  # %
        }

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}

# Legacy alias
@app.post("/detect")
async def detect_alias(image: UploadFile = File(...), conf: float = 0.25, imgsz: int = 320):
    return await detect(image=image, conf=conf, imgsz=imgsz)

# Stubs to avoid 404s from the frontend
@app.get("/api/history")
def api_history(page: int = 1, limit: int = 10):
    return {"total": 0, "data": []}

@app.get("/history")
def history_alias(page: int = 1, limit: int = 10):
    return api_history(page=page, limit=limit)

# local-run fallback
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
