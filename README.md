# 🫁 PneumoScan — Pneumonia Detection Web App

A full-stack web application for detecting pneumonia from chest X-ray images using your deep learning model.

---

## 📁 Project Structure

```
pneumonia-app/
├── backend/
│   ├── main.py           ← FastAPI server + model inference
│   └── requirements.txt
└── frontend/
    └── src/
        └── App.jsx       ← React UI (upload, results, history)
```

---

## 🚀 Setup Guide

### Step 1 — Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

**Connect your model** — open `main.py` and update line 21:
```python
MODEL_PATH = "pneumonia_model.h5"   # ← path to your saved model file
```
Supports **TensorFlow/Keras** (`.h5`) out of the box. For PyTorch, uncomment the PyTorch block in `load_model()`.

**Start the server:**
```bash
uvicorn main:app --reload --port 8000
```

Test it: http://localhost:8000/health

---

### Step 2 — Frontend Setup

```bash
cd frontend
npx create-react-app . --template minimal   # or: npm create vite@latest . -- --template react
# then replace src/App.js with src/App.jsx
npm install
npm start
```

The UI opens at http://localhost:3000

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check server + model status |
| POST | `/predict` | Upload image → get prediction |
| GET | `/history` | Fetch all past scans |
| DELETE | `/history/{id}` | Delete a scan record |

---

## 🧠 Model Output Format

### Binary sigmoid (recommended):
```
output shape: (1, 1)
0.0 → NORMAL, 1.0 → PNEUMONIA
```

### Softmax:
```
output shape: (1, 2)
[normal_prob, pneumonia_prob]
```

Both are handled automatically in `run_inference()`.

---

## ☁️ Deploying Your Model (from Google Colab)

If your model is still in Google Colab, export it first:

```python
# In Colab — save your model
model.save("pneumonia_model.h5")

# Download it
from google.colab import files
files.download("pneumonia_model.h5")
```

Then place `pneumonia_model.h5` in the `backend/` folder.

---

## 🌐 Production Deployment

| Layer | Free Option |
|-------|-------------|
| Backend | Render.com / Railway / Fly.io |
| Frontend | Vercel / Netlify |
| Model hosting | Hugging Face Spaces |

---

## ⚠️ Disclaimer

This tool is for **screening purposes only** and is not a substitute for professional medical diagnosis.
