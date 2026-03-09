import os
import io
import uuid
import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pydantic

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# We will create static directory
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = 'best_pneumonia_model.h5'

try:
    print("Loading model...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# In-memory history
history_db = []

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
    
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, 0)
        
        # Predict
        pred = model.predict(img_array)[0][0]
        
        is_pneumonia = bool(pred > 0.5)
        prediction = "PNEUMONIA" if is_pneumonia else "NORMAL"
        
        # Since pred is prob for class 1 (PNEUMONIA)
        confidence = float(pred * 100) if is_pneumonia else float((1.0 - pred) * 100)
        
        result = {
            "id": str(uuid.uuid4()),
            "filename": file.filename,
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "status": "Analyzed successfully",
            "timestamp": datetime.datetime.now().isoformat() + "Z"
        }
        
        # Add to history
        history_db.append(result)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/history")
async def get_history():
    # Return reversed history (latest first)
    return history_db[::-1]

@app.delete("/history/{id}")
async def delete_history(id: str):
    global history_db
    history_db = [h for h in history_db if h["id"] != id]
    return {"status": "deleted"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
