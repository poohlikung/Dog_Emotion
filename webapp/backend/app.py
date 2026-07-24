"""FastAPI app serving the dog-emotion classifier and the static frontend."""
import io
import os
from pathlib import Path

import jwt
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
from torchvision import models, transforms
from torch import nn

BACKEND_DIR = Path(__file__).resolve().parent
STATIC_DIR = BACKEND_DIR.parent / "static"
MODEL_PATH = BACKEND_DIR / "model.pt"

# --- History (JWT-in-cookie) config -------------------------------------------
# The visitor's prediction tally is stored inside a signed JWT kept in an
# httpOnly cookie, so history survives across visits with no database needed.
JWT_SECRET = os.environ.get("JWT_SECRET", "dog-emotion-dev-secret-change-me")
JWT_ALG = "HS256"
COOKIE_NAME = "dog_history"
COOKIE_MAX_AGE = 60 * 60 * 24 * 365  # 1 year


def encode_history(happy: int, angry: int) -> str:
    return jwt.encode({"h": happy, "a": angry}, JWT_SECRET, algorithm=JWT_ALG)


def decode_history(token: str | None):
    """Return {'h': int, 'a': int} if the cookie is present and valid, else None."""
    if not token:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return {"h": int(payload.get("h", 0)), "a": int(payload.get("a", 0))}
    except jwt.InvalidTokenError:
        return None


def history_summary(data) -> dict:
    """Build the trend summary the frontend renders from raw {'h','a'} counts."""
    happy, angry = data["h"], data["a"]
    total = happy + angry
    if total == 0:
        majority = None
    elif happy > angry:
        majority = "happy"
    elif angry > happy:
        majority = "angry"
    else:
        majority = "tie"
    return {"happy": happy, "angry": angry, "total": total, "majority": majority}


def set_history_cookie(response: Response, happy: int, angry: int) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=encode_history(happy, angry),
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        path="/",
    )

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


@app.post("/api/consent")
def grant_consent(request: Request, response: Response):
    """User agreed to storing history — start (or keep) their tally cookie."""
    data = decode_history(request.cookies.get(COOKIE_NAME)) or {"h": 0, "a": 0}
    set_history_cookie(response, data["h"], data["a"])
    return {"consent": True, "history": history_summary(data)}


@app.delete("/api/consent")
def revoke_consent(response: Response):
    """User withdrew consent — forget their history."""
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"consent": False}


@app.get("/api/history")
def get_history(request: Request):
    data = decode_history(request.cookies.get(COOKIE_NAME))
    return {
        "consent": data is not None,
        "history": history_summary(data) if data is not None else None,
    }


@app.post("/api/predict")
async def predict(request: Request, response: Response, file: UploadFile = File(...)):
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

    # If the visitor consented (cookie present), fold this result into their tally.
    history = None
    data = decode_history(request.cookies.get(COOKIE_NAME))
    if data is not None:
        if label == "happy":
            data["h"] += 1
        else:
            data["a"] += 1
        set_history_cookie(response, data["h"], data["a"])
        history = history_summary(data)

    return {
        "label": label,
        "confidence": probs[label],
        "probabilities": probs,
        "message": meta["message"],
        "tips": meta["tips"],
        "history": history,
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
