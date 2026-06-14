"""
make_reference_sheet.py — Sign Language Reference Sheet
========================================================
Scans dataset/images/, picks the best (clearest, preferably right-hand)
sample for each label, and saves a single reference grid image.

Output: reference_sheet.png  (next to this script)

Usage:
    python make_reference_sheet.py
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode
)
import numpy as np
import os
import sys
import math
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "dataset" / "images"
TASK_PATH  = BASE_DIR / "hand_landmarker.task"
OUTPUT     = BASE_DIR / "reference_sheet.png"

# ── Layout ────────────────────────────────────────────────────────────────────
CELL_W       = 300          # width of each sign cell (px)
CELL_H       = 300          # height of the image portion
LABEL_H      = 50           # height of label bar below image
PAD          = 16           # gap between cells
COLS         = 6            # signs per row (adjust freely)
BG_COLOR     = (30, 30, 30)
CELL_BG      = (50, 50, 50)
LABEL_BG     = (20, 20, 20)
LABEL_COLOR  = (255, 255, 255)
BORDER_COLOR = (90, 90, 90)
FONT         = cv2.FONT_HERSHEY_SIMPLEX

# ── MediaPipe detector (used to score each candidate image) ───────────────────
def create_detector():
    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(TASK_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,   # low threshold — we want a score
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return HandLandmarker.create_from_options(options)


def score_image(detector, img_bgr):
    """
    Returns (score, is_right_hand).
    score combines detection confidence + a penalty when landmarks are
    near the image border (clipped hand).
    """
    rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    if not result.hand_landmarks:
        return 0.0, False

    is_right = False
    if result.handedness:
        label = result.handedness[0][0].category_name
        is_right = (label == "Right")

    conf = result.handedness[0][0].score if result.handedness else 0.5

    # penalise images where any landmark is within 5% of the image edge
    # (hand is clipped) — subtract up to 0.4 from the score
    lms   = result.hand_landmarks[0]
    margin = 0.05
    clip_count = sum(
        1 for lm in lms
        if lm.x < margin or lm.x > 1 - margin
        or lm.y < margin or lm.y > 1 - margin
    )
    clip_penalty = min(0.4, clip_count * 0.04)

    return float(conf - clip_penalty), is_right


def best_sample(detector, paths):
    """
    Given a list of image paths for one label, return the path of the
    best one: prefer right-hand, break ties by confidence score.
    """
    scored = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        score, is_right = score_image(detector, img)
        # right-hand bonus so it wins ties clearly
        scored.append((score + (0.5 if is_right else 0.0), is_right, score, p))

    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][3]


def make_cell(img_bgr, inner_pad=16):
    """
    Letterbox (contain) the image inside CELL_W x CELL_H with inner_pad
    margin on all sides so the hand is never clipped or touching the edge.
    Empty space is filled with CELL_BG.
    """
    target_w = CELL_W - 2 * inner_pad
    target_h = CELL_H - 2 * inner_pad

    h, w = img_bgr.shape[:2]
    # scale so the LONGER side fits — whole image always visible
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    cell = np.full((CELL_H, CELL_W, 3), CELL_BG, dtype=np.uint8)
    # centre the resized image inside the cell
    y0 = (CELL_H - nh) // 2
    x0 = (CELL_W - nw) // 2
    cell[y0:y0+nh, x0:x0+nw] = resized
    return cell


def draw_label(canvas, text, x, y, w, h):
    """Draw label centred in the label bar."""
    # background
    cv2.rectangle(canvas, (x, y), (x+w, y+h), LABEL_BG, -1)
    # choose font scale based on label length
    fs   = 1.0 if len(text) <= 2 else 0.75
    thk  = 2
    (tw, th), _ = cv2.getTextSize(text, FONT, fs, thk)
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2
    cv2.putText(canvas, text, (tx, ty), FONT, fs, LABEL_COLOR, thk, cv2.LINE_AA)


def main():
    if not IMAGES_DIR.exists():
        print(f"ERROR: images folder not found at {IMAGES_DIR}")
        sys.exit(1)
    if not TASK_PATH.exists():
        print(f"ERROR: hand_landmarker.task not found at {TASK_PATH}")
        sys.exit(1)

    print("Scanning dataset/images/ ...")
    # group files by label (label is prefix before first '_')
    from collections import defaultdict
    label_files = defaultdict(list)
    for f in sorted(IMAGES_DIR.iterdir()):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        label = f.stem.split("_")[0]
        label_files[label].append(f)

    labels = sorted(label_files.keys())
    print(f"Found {len(labels)} labels: {labels}")

    detector = create_detector()

    # ── Pick best image per label ─────────────────────────────────────────────
    best = {}
    for label in labels:
        candidates = label_files[label]
        print(f"  [{label}]  {len(candidates)} candidates ...", end=" ", flush=True)
        path = best_sample(detector, candidates)
        if path:
            best[label] = path
            print(f"-> {path.name}")
        else:
            print("no valid image found, skipping")

    detector.close()

    if not best:
        print("No valid images found. Exiting.")
        sys.exit(1)

    labels_to_draw = sorted(best.keys())
    n      = len(labels_to_draw)
    cols   = min(COLS, n)
    rows   = math.ceil(n / cols)

    total_w = cols * CELL_W + (cols + 1) * PAD
    total_h = rows * (CELL_H + LABEL_H) + (rows + 1) * PAD

    canvas = np.full((total_h, total_w, 3), BG_COLOR, dtype=np.uint8)

    for idx, label in enumerate(labels_to_draw):
        row = idx // cols
        col = idx %  cols

        x = PAD + col * (CELL_W + PAD)
        y = PAD + row * (CELL_H + LABEL_H + PAD)

        # cell background
        cv2.rectangle(canvas, (x, y), (x+CELL_W, y+CELL_H+LABEL_H), CELL_BG, -1)

        # image
        img = cv2.imread(str(best[label]))
        if img is not None:
            cell = make_cell(img)
            canvas[y:y+CELL_H, x:x+CELL_W] = cell

        # border
        cv2.rectangle(canvas, (x, y), (x+CELL_W, y+CELL_H+LABEL_H),
                      BORDER_COLOR, 1)

        # label bar
        draw_label(canvas, label, x, y+CELL_H, CELL_W, LABEL_H)

    # ── title bar ─────────────────────────────────────────────────────────────
    title_h = 50
    title_canvas = np.full((title_h, total_w, 3), (20, 20, 20), dtype=np.uint8)
    title = f"Partial Sign Language Reference Sheet  ({n} signs)"
    (tw, th), _ = cv2.getTextSize(title, FONT, 0.75, 2)
    cv2.putText(title_canvas, title,
                ((total_w - tw)//2, (title_h + th)//2),
                FONT, 0.75, (200, 200, 200), 2, cv2.LINE_AA)
    canvas = np.vstack([title_canvas, canvas])

    cv2.imwrite(str(OUTPUT), canvas)
    print(f"\n✓ Reference sheet saved -> {OUTPUT}")
    print(f"  Grid: {cols} cols x {rows} rows  |  Image size: {canvas.shape[1]}x{canvas.shape[0]}px")

    # show it
    cv2.imshow("Sign Language Reference Sheet  (any key to close)", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()