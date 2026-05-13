from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import uvicorn
import numpy as np
from PIL import Image
import io
import base64

from keras.models import load_model

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

model = load_model("classifier.keras")
labels = ['No Fire Detected', 'Fire Detected']


# -----------------------------
# Request Model
# -----------------------------
class PredictRequest(BaseModel):
    image: str


# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))

    img = img.convert("RGB")
    img = img.resize((32, 32))

    img = np.array(img)
    img = img / 255.0

    img = img.reshape(1, 32, 32, 3).astype(np.float32)

    return img


# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/about", response_class=HTMLResponse)
def about():
    with open("static/about.html", "r", encoding="utf-8") as f:
        return f.read()


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
async def predict(data: PredictRequest):

    try:
        # Decode base64 image
        image_bytes = base64.b64decode(data.image)

        # Preprocess image
        img_tensor = preprocess_image(image_bytes)

        # Predict
        prediction = model.predict(img_tensor)

        confidence = float(np.max(prediction)) * 100
        label_index = int(np.argmax(prediction))

        detected = labels[label_index] == "Fire Detected"

        return {
            "detected": detected,
            "label": labels[label_index],
            "confidence": round(confidence, 2),
            "summary": "Wildfire detected in image."
            if detected
            else "No wildfire detected in image."
        }

    except Exception as e:
        return {
            "error": str(e)
        }


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)