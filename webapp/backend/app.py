"""FastAPI app serving the dog-emotion classifier and the static frontend."""
import io
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
from torchvision import models, transforms
from torch import nn

BACKEND_DIR = Path(__file__).resolve().parent
STATIC_DIR = BACKEND_DIR.parent / "static"
MODEL_PATH = BACKEND_DIR / "model.pt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EMOTION_META = {
    "happy": {
        "message": "หมาของคุณดูมีความสุขและผ่อนคลาย!",
        "tips": [
            "พาไปเดินเล่นหรือเล่นของเล่นที่ชอบ เพื่อรักษาอารมณ์ดีนี้ไว้",
            "ให้รางวัลเมื่อแสดงพฤติกรรมดี ๆ",
            "รักษากิจวัตรประจำวันให้สม่ำเสมอ",
        ],
    },
    "angry": {
        "message": "หมาของคุณอาจกำลังเครียดหรือไม่พอใจอยู่",
        "tips": [
            "หลีกเลี่ยงสิ่งกระตุ้น เช่น เสียงดังหรือการเผชิญหน้ากับสัตว์อื่น",
            "ให้พื้นที่และเวลาให้เขาสงบลง อย่าเข้าใกล้กะทันหัน",
            "ฝึกคำสั่งง่าย ๆ เช่น 'นั่ง' เพื่อช่วยควบคุมอารมณ์",
        ],
    },
}


class EmotionClassifier:
    def __init__(self, model_path: Path):
        checkpoint = torch.load(model_path, map_location="cpu")
        self.classes = checkpoint["classes"]
        self.img_size = checkpoint["img_size"]

        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.last_channel, len(self.classes)),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        self.model = model

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    @torch.no_grad()
    def predict(self, image: Image.Image):
        x = self.transform(image.convert("RGB")).unsqueeze(0)
        logits = self.model(x)[0]
        probs = F.softmax(logits, dim=0).tolist()
        return {cls: round(p, 4) for cls, p in zip(self.classes, probs)}


app = FastAPI(title="Dog Emotion Detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier: EmotionClassifier | None = None
if MODEL_PATH.exists():
    classifier = EmotionClassifier(MODEL_PATH)


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": classifier is not None}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read that image file")

    probs = classifier.predict(image)
    label = max(probs, key=probs.get)
    meta = EMOTION_META[label]

    return {
        "label": label,
        "confidence": probs[label],
        "probabilities": probs,
        "message": meta["message"],
        "tips": meta["tips"],
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
