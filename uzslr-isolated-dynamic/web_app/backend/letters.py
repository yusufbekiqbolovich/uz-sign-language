"""
letters.py — Fingerspelling (letters & numbers) recognition
============================================================
Static single-frame recognition using a scikit-learn classifier on
MediaPipe HandLandmarker keypoints. Ported from the standalone Flask
app (../../../app.py) so the merged FastAPI server can serve both the
word model (torch) and the letter model (sklearn) from one process.

Assets live in the parent fingerspelling repo:
    <fingerspelling>/model.pkl
    <fingerspelling>/hand_landmarker.task
    <fingerspelling>/dataset/images/
"""

import base64
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode,
)

from web_app.backend.config import (
    LETTERS_MODEL_PATH, HAND_TASK_PATH, LETTERS_IMAGES_DIR,
)


# module-level singletons, populated by load()
_clf = None
_le = None
_detector = None
_label_images: dict[str, list[str]] = defaultdict(list)


def normalize_landmarks(landmarks):
    """Wrist-centred, scale-normalised 63-dim feature vector (must match training)."""
    lm = np.array([[l.x, l.y, l.z] for l in landmarks], dtype=np.float32)
    lm -= lm[0].copy()
    scale = np.linalg.norm(lm[9])
    if scale > 1e-6:
        lm /= scale
    return lm.flatten()


def _build_index():
    images_dir = Path(LETTERS_IMAGES_DIR)
    if not images_dir.exists():
        return
    for f in sorted(images_dir.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            label = f.stem.split("_")[0]
            _label_images[label].append(f.name)


def load():
    """Load model + detector + image index. Call once at startup."""
    global _clf, _le, _detector

    model_path = Path(LETTERS_MODEL_PATH)
    task_path = Path(HAND_TASK_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"letters model not found: {model_path}")
    if not task_path.exists():
        raise FileNotFoundError(f"hand_landmarker.task not found: {task_path}")

    payload = joblib.load(str(model_path))
    _clf = payload["model"]
    _le = payload["label_encoder"]

    _detector = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(task_path)),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    )

    _build_index()
    print(f"[startup] letters model loaded. classes: {list(_le.classes_)}")
    print(f"[startup] letter image index built: {len(_label_images)} labels")


def labels() -> list[dict]:
    """All labels with their sample counts, sorted."""
    return [
        {"label": l, "count": len(_label_images[l])}
        for l in sorted(_label_images.keys())
    ]


def images(label: str, n: int = 6) -> list[str]:
    """Up to N evenly-spaced sample filenames for a label."""
    files = _label_images.get(label.upper(), [])
    if len(files) <= n:
        return files
    step = len(files) / n
    return [files[int(i * step)] for i in range(n)]


def predict(image_b64: str) -> dict:
    """Decode a base64 JPEG/PNG frame and return the predicted letter/number."""
    if not image_b64:
        return {"error": "no image"}

    _, encoded = image_b64.split(",", 1) if "," in image_b64 else ("", image_b64)
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "decode failed"}

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _detector.detect(mp_img)

    if not result.hand_landmarks:
        return {"label": None, "confidence": 0, "hand": False}

    lms = result.hand_landmarks[0]
    features = normalize_landmarks(lms).reshape(1, -1)
    proba = _clf.predict_proba(features)[0]
    idx = int(np.argmax(proba))
    label = str(_le.classes_[idx])
    conf = float(proba[idx])
    landmarks_xy = [[float(lm.x), float(lm.y)] for lm in lms]
    return {
        "label": label,
        "confidence": round(conf, 3),
        "hand": True,
        "landmarks": landmarks_xy,
    }
