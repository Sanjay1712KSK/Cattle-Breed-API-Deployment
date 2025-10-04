import os
import requests

MODEL_PATH = "cattle_breed_mobilenetv2_v5promax_finetuned.keras"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1MhJpvwnBOtkOp-_XkjuFTrLVXGvJduCC"  # Replace with your Google Drive file ID

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("Model downloaded successfully!")












from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import json
import time

# ✅ Import breed descriptions
from Normal_Detection.breed_descriptions import breed_descriptions

# -------------------------
# 1. Load model & class indices
# -------------------------
MODEL_PATH = "cattle_breed_mobilenetv2_v5promax_finetuned.keras"
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully:", MODEL_PATH)

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

# -------------------------
# 2. Initialize FastAPI
# -------------------------
app = FastAPI(title="Cattle Breed Recognition API")

# -------------------------
# 3. Image preprocessing
# -------------------------
IMG_HEIGHT, IMG_WIDTH = 256, 256

def preprocess_image(file_bytes):
    img = image.load_img(io.BytesIO(file_bytes), target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------
# 4. Smart text generator
# -------------------------
def generate_text_response(breed_name, confidence, description):
    confidence_percent = round(confidence * 100, 2)
    if confidence > 0.85:
        confidence_text = f"I am highly confident ({confidence_percent}%)"
    elif confidence > 0.6:
        confidence_text = f"I am fairly confident ({confidence_percent}%)"
    else:
        confidence_text = f"The prediction is uncertain ({confidence_percent}%)"

    return (
        f"{confidence_text} that this is a **{breed_name}**.\n\n"
        f"📖 About this breed:\n{description}"
    )

# -------------------------
# 5. Normal Upload Endpoint
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_array = preprocess_image(contents)
        preds = model.predict(img_array)
        class_idx = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        breed_name = idx_to_class[class_idx]
        description = breed_descriptions.get(breed_name, "No description available")

        message = generate_text_response(breed_name, confidence, description)

        return JSONResponse({
            "class": breed_name,
            "confidence": confidence,
            "description": description,
            "message": message
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# -------------------------
# 6. Real-Time Detection Endpoint
# -------------------------
@app.post("/predict-frame")
async def predict_frame(file: UploadFile = File(...)):
    """
    Optimized endpoint for real-time camera feed.
    Returns only class and confidence (faster).
    """
    try:
        start = time.time()
        contents = await file.read()
        img_array = preprocess_image(contents)
        preds = model.predict(img_array)
        class_idx = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        breed_name = idx_to_class[class_idx]
        end = time.time()

        return JSONResponse({
            "class": breed_name,
            "confidence": confidence,
            "latency_ms": round((end - start) * 1000, 2)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# -------------------------
# 7. Root Endpoint
# -------------------------
@app.get("/")
async def root():
    return {"message": "Cattle Breed Recognition API is running!"}
