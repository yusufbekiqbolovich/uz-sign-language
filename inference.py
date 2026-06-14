"""
inference.py — Live Sign Language Detection
============================================
Uses MediaPipe Tasks API (mediapipe 0.10.30+)

Controls:  Q → quit
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import numpy as np
import joblib
import os
import sys
import time
from collections import deque, Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
TASK_PATH  = os.path.join(BASE_DIR, "hand_landmarker.task")

SMOOTH_WINDOW  = 10
CONF_THRESHOLD = 0.65

# ── MediaPipe Tasks detector ──────────────────────────────────────────────────
def create_detector():
    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=TASK_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.8,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return HandLandmarker.create_from_options(options)

# ── Hand skeleton drawing ─────────────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts  = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 5, (255, 255, 255), -1)
        cv2.circle(frame, pt, 5, (0, 150, 0), 1)

def bounding_box(frame, landmarks, pad=20):
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1 = max(0, int(min(xs))-pad);  y1 = max(0, int(min(ys))-pad)
    x2 = min(w, int(max(xs))+pad);  y2 = min(h, int(max(ys))+pad)
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,220,0), 2)
    return x1, y1

# ── Landmark normalisation (identical to collect_data.py) ─────────────────────
def normalize_landmarks(landmarks):
    lm    = np.array([[l.x, l.y, l.z] for l in landmarks], dtype=np.float32)
    lm   -= lm[0].copy()
    scale = np.linalg.norm(lm[9])
    if scale > 1e-6:
        lm /= scale
    return lm.flatten()

# ── Smoothing ─────────────────────────────────────────────────────────────────
def majority_vote(history):
    if not history:
        return None, 0.0
    counts = Counter(history)
    label, cnt = counts.most_common(1)[0]
    return label, cnt / len(history)

# ── Drawing ───────────────────────────────────────────────────────────────────
FONT  = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255,255,255); BLACK = (0,0,0)
GREEN = (0,220,0);     RED   = (0,0,220)

def conf_color(c):
    if c >= 0.85: return (0, 210, 0)
    if c >= 0.65: return (0, 200, 200)
    return (0, 80, 210)

def draw_prediction(frame, label, conf, x1, y1):
    color = conf_color(conf)
    # shadow then label
    cv2.putText(frame, label, (x1+3, y1-10), FONT, 2.5, BLACK, 6)
    cv2.putText(frame, label, (x1,   y1-10), FONT, 2.5, color, 4)
    cv2.putText(frame, f"{conf*100:.0f}%", (x1+90, y1-10), FONT, 1.0, color, 2)

def draw_hud(frame, hand_ok, smoothed, s_conf, fps):
    h, w = frame.shape[:2]
    ov   = frame.copy()
    cv2.rectangle(ov, (0,0), (w,52), BLACK, -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "Sign Language - Live Detection", (10,34), FONT, 0.9, WHITE, 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w-110, 34), FONT, 0.8, WHITE, 1)

    col = GREEN if hand_ok else RED
    cv2.circle(frame, (w-20, 72), 10, col, -1)
    cv2.putText(frame, "Hand OK" if hand_ok else "No hand",
                (w-140, 76), FONT, 0.6, col, 1)

    if smoothed and s_conf >= CONF_THRESHOLD:
        panel = frame.copy()
        cv2.rectangle(panel, (0, h-72), (w, h), BLACK, -1)
        cv2.addWeighted(panel, 0.6, frame, 0.4, 0, frame)
        color = conf_color(s_conf)
        cv2.putText(frame, f"Prediction: {smoothed}",
                    (16, h-30), FONT, 1.4, color, 3)
        cv2.putText(frame, f"Confidence: {s_conf*100:.0f}%",
                    (16, h-8),  FONT, 0.7, color, 2)
    elif not hand_ok:
        cv2.putText(frame, "Show your right hand ...",
                    (10, h-20), FONT, 0.8, WHITE, 1)

    cv2.putText(frame, "Q = quit", (w-105, h-14), FONT, 0.55, WHITE, 1)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*55)
    print("  Sign Language — Live Inference  (mediapipe 0.10+)")
    print("="*55)

    for path, name in [(MODEL_PATH, "model.pkl"), (TASK_PATH, "hand_landmarker.task")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at {path}")
            tip = "Run train_model.py first." if "model" in name \
                  else "Run collect_data.py first (it downloads the task file)."
            print(tip);  sys.exit(1)

    payload   = joblib.load(MODEL_PATH)
    model, le = payload["model"], payload["label_encoder"]
    print(f"✓ Model loaded. Classes: {list(le.classes_)}")

    detector  = create_detector()
    cap       = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.");  sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    label_hist = deque(maxlen=SMOOTH_WINDOW)
    conf_hist  = deque(maxlen=SMOOTH_WINDOW)
    prev_t     = time.time();  fps = 0.0

    print("Webcam open — press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector.detect(mp_img)

        now    = time.time()
        fps    = 0.9 * fps + 0.1 / max(now - prev_t, 1e-6)
        prev_t = now

        hand_ok   = bool(result.hand_landmarks)
        smoothed  = None
        s_conf    = 0.0

        if hand_ok:
            lms      = result.hand_landmarks[0]
            features = normalize_landmarks(lms).reshape(1, -1)
            proba    = model.predict_proba(features)[0]
            idx      = np.argmax(proba)
            raw_lbl  = str(le.classes_[idx])
            raw_conf = float(proba[idx])

            label_hist.append(raw_lbl)
            conf_hist.append(raw_conf)
            smoothed, _ = majority_vote(label_hist)
            s_conf      = float(np.mean(conf_hist))

            draw_hand(frame, lms)
            x1, y1 = bounding_box(frame, lms)
            if raw_conf >= CONF_THRESHOLD:
                draw_prediction(frame, raw_lbl, raw_conf, x1, y1)
        else:
            label_hist.clear();  conf_hist.clear()

        draw_hud(frame, hand_ok, smoothed, s_conf, fps)

        cv2.imshow("Sign Language — Live Detection", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Bye!")

if __name__ == "__main__":
    main()