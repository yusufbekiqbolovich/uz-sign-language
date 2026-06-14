"""
collect_data.py — Sign Language Data Collection
================================================
Uses MediaPipe Tasks API (mediapipe 0.10.30+)

Controls:
  SPACE  → capture current frame (landmarks + image)
  N      → done with current label, prompt for next
  Q      → quit and save everything

Workflow:
  1. Run the script → prompted for first label in terminal
  2. Do the sign → press SPACE as many times as you want
  3. Press N → prompted for next label
  4. Repeat until done → press Q to quit
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import numpy as np
import pandas as pd
import os
import sys
import urllib.request
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR  = os.path.join(DATASET_DIR, "images")
CSV_PATH    = os.path.join(DATASET_DIR, "landmarks.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "hand_landmarker.task")

os.makedirs(IMAGES_DIR, exist_ok=True)

# ── Download model if needed ──────────────────────────────────────────────────
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading hand_landmarker.task (~26 MB) …")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print(f"✓ Model saved → {MODEL_PATH}")
        except Exception as e:
            print(f"ERROR downloading model: {e}")
            print(f"Download manually from:\n  {MODEL_URL}")
            print(f"Save to: {MODEL_PATH}")
            sys.exit(1)

# ── MediaPipe Tasks detector ──────────────────────────────────────────────────
def create_detector():
    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.8,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return HandLandmarker.create_from_options(options)

# ── Draw skeleton ─────────────────────────────────────────────────────────────
# Hand connection pairs (same topology as the old API)
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (5,9),(9,10),(10,11),(11,12),     # middle
    (9,13),(13,14),(14,15),(15,16),   # ring
    (13,17),(17,18),(18,19),(19,20),  # pinky
    (0,17),                           # palm base
]

def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts  = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 5, (255, 255, 255), -1)
        cv2.circle(frame, pt, 5, (0, 150, 0), 1)

# ── Landmark normalisation ────────────────────────────────────────────────────
def normalize_landmarks(landmarks):
    """
    Translate wrist (lm 0) to origin, scale by wrist→middle-MCP (lm 9).
    Returns flat array of 63 floats (21 × x, y, z).
    """
    lm    = np.array([[l.x, l.y, l.z] for l in landmarks], dtype=np.float32)
    lm   -= lm[0].copy()                      # translate
    scale = np.linalg.norm(lm[9])
    if scale > 1e-6:
        lm /= scale
    return lm.flatten()

# ── CSV helpers ───────────────────────────────────────────────────────────────
FEAT_COLS = [f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")]
COLUMNS   = ["label"] + FEAT_COLS

def load_csv():
    return pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) \
           else pd.DataFrame(columns=COLUMNS)

def append_row(df, label, features):
    row = {"label": label}
    for j, val in enumerate(features):
        row[f"{'xyz'[j%3]}{j//3}"] = val
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

# ── HUD drawing ───────────────────────────────────────────────────────────────
FONT  = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 220, 0);  RED   = (0, 0, 220)
WHITE = (255,255,255); BLACK = (0,0,0); CYAN = (220,220,0)

def draw_hud(frame, label, count, total, hand_ok, flash):
    h, w = frame.shape[:2]
    ov   = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 52), BLACK, -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"Label: [{label}]",        (10, 34),     FONT, 1.0, CYAN,  2)
    cv2.putText(frame, f"Captured: {count}",        (w-220, 34),  FONT, 0.9, WHITE, 2)
    cv2.putText(frame, f"Total saved: {total}",     (10, h-14),   FONT, 0.6, WHITE, 1)
    cv2.putText(frame, "SPACE=capture  N=next  Q=quit",
                (w-350, h-14), FONT, 0.52, WHITE, 1)

    col = GREEN if hand_ok else RED
    cv2.circle(frame, (w-20, 20), 10, col, -1)
    cv2.putText(frame, "Hand OK" if hand_ok else "No hand",
                (w-140, 24), FONT, 0.6, col, 1)

    if flash:
        cv2.rectangle(frame, (0,0), (w,h), GREEN, 6)

def crop_hand(frame, landmarks, pad=30):
    h, w = frame.shape[:2]
    xs   = [lm.x * w for lm in landmarks]
    ys   = [lm.y * h for lm in landmarks]
    x1   = max(0, int(min(xs))-pad);  y1 = max(0, int(min(ys))-pad)
    x2   = min(w, int(max(xs))+pad);  y2 = min(h, int(max(ys))+pad)
    return frame[y1:y2, x1:x2]

# ── Main ──────────────────────────────────────────────────────────────────────
def get_label(prompt):
    sys.stdout.write(f"\n{prompt}");  sys.stdout.flush()
    v = input().strip().upper()
    return v or None

def main():
    print("="*55)
    print("  Sign Language Data Collector  (mediapipe 0.10+)")
    print("="*55)
    print("  SPACE → capture  |  N → next label  |  Q → quit")
    print("="*55)

    ensure_model()
    detector = create_detector()
    df       = load_csv()

    label = get_label("Enter first label (e.g. A, B, 1): ")
    if not label:
        print("No label — exiting.");  return

    count   = 0
    total   = len(df)
    flash   = 0

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.");  return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"\n[ACTIVE] '{label}'  — SPACE=capture, N=next, Q=quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame    = cv2.flip(frame, 1)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result   = detector.detect(mp_img)

        hand_ok   = bool(result.hand_landmarks)
        landmarks = result.hand_landmarks[0] if hand_ok else None
        features  = normalize_landmarks(landmarks) if hand_ok else None

        if hand_ok:
            draw_hand(frame, landmarks)

        draw_hud(frame, label, count, total, hand_ok, flash > 0)
        if flash > 0:
            flash -= 1

        cv2.imshow("Sign Language — Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):                        # ── capture
            if not hand_ok:
                print("  [SKIP] No hand visible.")
            else:
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = os.path.join(IMAGES_DIR, f"{label}_{ts}.jpg")
                crop = crop_hand(frame, landmarks)
                if crop.size > 0:
                    cv2.imwrite(path, crop)
                df    = append_row(df, label, features)
                count += 1;  total += 1;  flash = 4
                print(f"  [SAVED] '{label}' #{count}  (total: {total})")

        elif key in (ord("n"), ord("N")):          # ── next label
            print(f"\n  Done '{label}' — {count} samples.")
            df.to_csv(CSV_PATH, index=False)
            new = get_label("Enter next label (Enter=quit): ")
            if not new:
                break
            label = new;  count = 0
            print(f"\n[ACTIVE] '{label}'  — SPACE=capture, N=next, Q=quit")

        elif key in (ord("q"), ord("Q")):          # ── quit
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    df.to_csv(CSV_PATH, index=False)

    print(f"\n✓ Saved → {CSV_PATH}  ({total} total samples)")
    if total > 0:
        print("\nSamples per label:")
        for lbl, cnt in df["label"].value_counts().items():
            print(f"  {lbl:>4}: {cnt:>3}  {'█'*cnt}")

if __name__ == "__main__":
    main()