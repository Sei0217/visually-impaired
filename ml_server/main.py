# ml_server/main.py
import io, base64, os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ---------- perf limits for small instances ----------
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
# -----------------------------------------------------

app = FastAPI()
# tighten if you like: replace "*" with your Vercel origin(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
model = YOLO(MODEL_PATH)

# ---------- startup warmup ----------
@app.on_event("startup")
def _warmup():
    try:
        img = Image.new("RGB", (64, 64), (0, 0, 0))
        if torch is not None:
            with torch.no_grad():
                model.predict(img, conf=0.25, imgsz=320)
        else:
            model.predict(img, conf=0.25, imgsz=320)
        print("[startup] warmup done")
    except Exception as e:
        print("[startup] warmup failed:", e)
# -----------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

def _load_font(px):
    """
    Try env FONT_PATH first; fallback to common system fonts; last resort default font.
    """
    candidates = []
    env = os.getenv("FONT_PATH")
    if env:
        candidates.append(env)
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, px)
        except Exception:
            continue
    # fallback (fixed small size)
    return ImageFont.load_default()

@app.post("/api/detect")
async def detect(
    image: UploadFile = File(...),
    conf: float = 0.25,
    imgsz: int = 320,          # faster default
    return_image: bool = True
):
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

    raw = await image.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    # inference
    if torch is not None:
        with torch.no_grad():
            r = model.predict(img, conf=conf, imgsz=imgsz)[0]
    else:
        r = model.predict(img, conf=conf, imgsz=imgsz)[0]

    boxes = r.boxes.xyxy.cpu().tolist()
    clss  = r.boxes.cls.cpu().tolist()
    confs = r.boxes.conf.cpu().tolist()
    names = r.names

    dets = []
    for (x1, y1, x2, y2), c, p in zip(boxes, clss, confs):
        dets.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "label": names[int(c)],
            "confidence": float(p)
        })

    payload = {"detections": dets}

    if return_image:
        W, H = img.size
        # dynamic styling
        line_w = max(2, int(min(W, H) * 0.006))        # box thickness
        font_px = max(16, int(min(W, H) * 0.045))      # label size (~4.5% of min dim)
        pad = max(4, int(font_px * 0.35))              # label padding

        font = _load_font(font_px)
        draw = ImageDraw.Draw(img)

        for dct in dets:
            x1, y1, x2, y2 = map(int, dct["bbox"])
            label_txt = f'{dct["label"]} — {int(round(dct["confidence"] * 100))}%'

            # box
            draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=line_w)

            # text size
            try:
                # precise size with provided font
                l, t, r, b = draw.textbbox((0, 0), label_txt, font=font)
                tw, th = (r - l), (b - t)
            except Exception:
                # fallback estimate
                tw, th = (len(label_txt) * max(8, font_px // 2), font_px)

            # position label above box if there's space; otherwise inside top-left
            top_y = y1 - (th + 2 * pad)
            if top_y < 0:
                top_y = y1 + line_w

            bg_left  = max(0, x1)
            bg_top   = max(0, top_y)
            bg_right = min(W, x1 + tw + 2 * pad)
            bg_bot   = min(H, top_y + th + 2 * pad)

            # label background (black) + text (white)
            draw.rectangle([(bg_left, bg_top), (bg_right, bg_bot)], fill=(0, 0, 0))
            draw.text((bg_left + pad, bg_top + pad), label_txt, font=font, fill=(255, 255, 255))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90, optimize=True)
        payload["image_base64"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    return payload

# optional local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
