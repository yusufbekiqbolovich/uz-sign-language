"""
app.py — Sign Language Practice UI (Flask backend)
===================================================
Run:
    python app.py
Then open http://localhost:5000 in your browser.

Serves:
  GET  /api/labels                    → list of all labels + sample counts
  GET  /api/images/<label>            → up to N image filenames for that label
  GET  /dataset/images/<filename>     → serve the actual image file
  POST /api/predict                   → receive a JPEG frame, return prediction
"""

import os, sys, cv2, json, base64, joblib, numpy as np
from pathlib import Path
from collections import defaultdict
from flask import Flask, jsonify, request, send_from_directory, send_file

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "dataset" / "images"
TASK_PATH  = BASE_DIR / "hand_landmarker.task"
MODEL_PATH = BASE_DIR / "model.pkl"

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")

# ── Load model ─────────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    print(f"ERROR: model.pkl not found. Run train_model.py first.")
    sys.exit(1)

payload   = joblib.load(MODEL_PATH)
clf       = payload["model"]
le        = payload["label_encoder"]
FEAT_COLS = payload["feature_cols"]
print(f"Model loaded. Classes: {list(le.classes_)}")

# ── MediaPipe detector ─────────────────────────────────────────────────────────
if not TASK_PATH.exists():
    print(f"ERROR: hand_landmarker.task not found. Run collect_data.py first.")
    sys.exit(1)

detector = HandLandmarker.create_from_options(
    HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(TASK_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
)

def normalize_landmarks(landmarks):
    lm  = np.array([[l.x, l.y, l.z] for l in landmarks], dtype=np.float32)
    lm -= lm[0].copy()
    scale = np.linalg.norm(lm[9])
    if scale > 1e-6:
        lm /= scale
    return lm.flatten()

# ── Image index ────────────────────────────────────────────────────────────────
label_images: dict[str, list[str]] = defaultdict(list)

def build_index():
    if not IMAGES_DIR.exists():
        return
    for f in sorted(IMAGES_DIR.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            label = f.stem.split("_")[0]
            label_images[label].append(f.name)
    print(f"Image index built: {len(label_images)} labels")

build_index()

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(BASE_DIR / "practice_ui.html")

@app.route("/hand_landmarker.task")
def serve_task():
    return send_file(TASK_PATH, mimetype="application/octet-stream")

@app.route("/api/labels")
def api_labels():
    labels = sorted(label_images.keys())
    return jsonify([
        {"label": l, "count": len(label_images[l])}
        for l in labels
    ])

@app.route("/api/images/<label>")
def api_images(label):
    files = label_images.get(label.upper(), [])
    # return up to 6 evenly-spaced samples so we get visual variety
    if len(files) <= 6:
        chosen = files
    else:
        step = len(files) / 6
        chosen = [files[int(i * step)] for i in range(6)]
    return jsonify(chosen)

@app.route("/dataset/images/<filename>")
def serve_image(filename):
    return send_from_directory(str(IMAGES_DIR), filename)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data    = request.get_json()
    img_b64 = data.get("image", "")
    if not img_b64:
        return jsonify({"error": "no image"}), 400

    # decode base64 JPEG/PNG from browser canvas
    header, encoded = img_b64.split(",", 1) if "," in img_b64 else ("", img_b64)
    img_bytes = base64.b64decode(encoded)
    arr       = np.frombuffer(img_bytes, np.uint8)
    frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "decode failed"}), 400

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    if not result.hand_landmarks:
        return jsonify({"label": None, "confidence": 0, "hand": False})

    lms      = result.hand_landmarks[0]
    features = normalize_landmarks(lms).reshape(1, -1)
    proba    = clf.predict_proba(features)[0]
    idx      = int(np.argmax(proba))
    label    = str(le.classes_[idx])
    conf     = float(proba[idx])
    landmarks_xy = [[float(lm.x), float(lm.y)] for lm in lms]
    return jsonify({"label": label, "confidence": round(conf, 3), "hand": True, "landmarks": landmarks_xy})

if __name__ == "__main__":
    print("Starting Sign Language Practice UI at http://localhost:5000")
    app.run(host="0.0.0.0", port=7860)