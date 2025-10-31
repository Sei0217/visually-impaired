# ml_server/main.py — patched
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np, cv2, os, traceback, base64

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Use model path relative to this file ===
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
print("MODEL_PATH:", MODEL_PATH, "exists:", os.path.exists(MODEL_PATH))

# Load once
model = YOLO(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

KEEP = {"person", "door", "stairs"}  # Adjust to your model's class names

@app.post("/api/detect")
async def detect(image: UploadFile = File(...)):
    try:
        data = await image.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"success": False, "error": "decode_failed"}

        # Inference
        r = model(img, imgsz=640, conf=0.25, verbose=False)[0]

        # Annotated image
        annotated = r.plot()  # BGR
        ok, buf = cv2.imencode(".jpg", annotated)
        if not ok:
            return {"success": False, "error": "encode_failed"}
        b64img = base64.b64encode(buf.tobytes()).decode("utf-8")
        data_url = "data:image/jpeg;base64," + b64img

        # Filter only 'person', 'stairs', 'door' class detections
        names = r.names if hasattr(r, "names") else model.names
        dets = []
        top_label = "No detection"
        top_conf = 0.0
        if r.boxes is not None and len(r.boxes) > 0:
            for xyxy, cls, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                label = names[int(cls)]
                if label in KEEP:  # Keep only 'person', 'stairs', 'door'
                    dets.append({"bbox": xyxy, "label": label, "conf": float(conf)})
            # Top-1
            if dets:
                top_idx = int(np.argmax([det["conf"] for det in dets]))
                top_label = dets[top_idx]["label"]
                top_conf = dets[top_idx]["conf"]

        return {
            "success": True,
            "annotated_image_url": data_url,
            "detections": dets,
            "object_type": top_label,
            "confidence_score": round(top_conf * 100, 2),
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# Alias to handle legacy or unhandled requests for /detect
@app.post("/detect")
async def detect_alias(image: UploadFile = File(...)):
    return await detect(image)

# History endpoint to prevent 404 from frontend requests (GET /api/history)
@app.get("/api/history")
def api_history(page: int = 1, limit: int = 10):
    return {"total": 0, "data": []}

# Alias for /history for frontend compatibility
@app.get("/history")
def history_alias(page: int = 1, limit: int = 10):
    return api_history(page=page, limit=limit)
