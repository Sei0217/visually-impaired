# ml_server/main.py
import io, base64, os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw
from ultralytics import YOLO

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
model = YOLO(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/detect")
async def detect(image: UploadFile = File(...), conf: float = 0.25, imgsz: int = 640, return_image: bool = True):
    raw = await image.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    r = model.predict(img, conf=conf, imgsz=imgsz)[0]
    boxes = r.boxes.xyxy.cpu().tolist()
    clss  = r.boxes.cls.cpu().tolist()
    confs = r.boxes.conf.cpu().tolist()
    names = r.names

    dets = []
    for (x1,y1,x2,y2), c, p in zip(boxes, clss, confs):
        dets.append({"bbox":[float(x1),float(y1),float(x2),float(y2)],
                     "label":names[int(c)], "confidence":float(p)})

    payload = {"detections": dets}

    if return_image:
        d = ImageDraw.Draw(img)
        for dct in dets:
            x1,y1,x2,y2 = dct["bbox"]
            label = f'{dct["label"]} {dct["confidence"]:.2f}'
            d.rectangle([(x1,y1),(x2,y2)], outline=(0,255,0), width=3)
            w = d.textlength(label); h = 18
            d.rectangle([(x1,y1-h-6),(x1+w+12,y1)], fill=(0,0,0))
            d.text((x1+6,y1-h-3), label, fill=(255,255,255))
        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=85)
        payload["image_base64"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    return payload
