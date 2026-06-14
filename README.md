# Sign Language Fingerspelling Recognition

A lightweight, webcam-based pipeline for recognising static hand signs — letters, numbers, or any custom gestures you define. No cloud, no GPU required. Runs in real time on a laptop.

Built with [MediaPipe](https://developers.google.com/mediapipe) for hand tracking and a scikit-learn Random Forest + MLP ensemble for classification.

---

## How it works

Each time you make a sign, MediaPipe detects 21 landmarks on your hand (x, y, z coordinates). These 63 values are normalised so the result is independent of your hand size, distance from the camera, and position in the frame. The classifier is trained on these normalised vectors — not on raw images — which is why it is fast, lightweight, and works across different people and lighting conditions.

```
Webcam frame
    └─► MediaPipe HandLandmarker  →  21 landmarks (x, y, z)
            └─► Normalise (wrist to origin, scale by palm size)
                    └─► 63-float feature vector
                            └─► Classifier  →  predicted label
```

---

## Project structure

```
sign_language/
├── collect_data.py         # Step 1 — capture labelled training samples
├── train_model.py          # Step 2 — train and evaluate the classifier
├── inference.py            # Step 3 — live webcam prediction
├── app.py                  # Step 5 — interactive practice UI (browser)
├── practice_ui.html        # Frontend for the practice UI (served by app.py)
├── make_reference_sheet.py # Optional — generate a visual reference grid
├── hand_landmarker.task    # MediaPipe model file
├── model.pkl               # Trained classifier (created by train_model.py)
├── reference_sheet.png     # Reference grid image (created by make_reference_sheet.py)
├── requirements.txt        # Python dependencies
├── README.md
└── dataset/
    ├── landmarks.csv       # Normalised landmark vectors + labels (your training data)
    └── images/             # Cropped hand images saved alongside each capture
```

---

## Setup

Requires Python 3.9 or later. Tested on macOS Apple Silicon, works on Windows and Linux too.

```bash
# Clone the repository
git clone https://github.com/yusufbekiqbolovich/uz-sign-language
cd uz-sign-language

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Optional: install latest dependency OR
pip install opencv-python mediapipe scikit-learn pandas 
numpy joblib

# Recommended: install exact dependecy versions
pip install -r requirements.txt
```

> **Apple Silicon note:** MediaPipe installs and runs natively via pip — no conda or Rosetta needed. If you hit any issues run `pip install mediapipe --upgrade`.

---

## Workflow

### Step 1 — Collect training data

```bash
python collect_data.py
```

The terminal will prompt you to enter a label. Type anything — a letter (`A`), a word (`HELLO`), a number (`1`), whatever sign you want to teach. Then:

- Hold your hand in front of the webcam — a green skeleton overlay confirms detection
- Press **`SPACE`** to capture the current frame (aim for **50–100 captures per sign**)
- Vary your hand angle and distance slightly between captures for better generalisation
- Press **`N`** when you are done with that sign — you will be prompted for the next label
- Press **`Q`** to quit and save everything

Each capture saves two things:
- A row of 63 normalised landmark floats + the label into `dataset/landmarks.csv`
- A cropped image of your hand into `dataset/images/` (useful for debugging and the reference sheet)

> **Tips for good data**
> - Capture with the hand that you will use during inference (right hand is the default preference)
> - Slightly vary the angle — front-on, a little left, a little right — across captures
> - Avoid capturing when the confidence indicator in the top-right corner shows red
> - Aim for at least 50 samples per sign; 100 gives noticeably better robustness

---

### Step 2 — Generate a reference sheet (optional but recommended)

```bash
python make_reference_sheet.py
```

Scans `dataset/images/`, picks the clearest capture for each label (preferring right-hand samples and penalising clipped hands), and produces a single `reference_sheet.png` grid — one cell per sign with the label below it. Use this as a cheat sheet while you are still learning or extending your dataset.

The output also pops up on screen immediately. Close it with any key.

---

### Step 3 — Train the classifier

```bash
python train_model.py
```

Reads `dataset/landmarks.csv`, trains a soft-voting ensemble of a Random Forest and an MLP, runs 5-fold cross-validation, and prints a full classification report and confusion matrix. The final model is retrained on the entire dataset and saved to `model.pkl`.

Training takes a few seconds even with thousands of samples. You will see output like:

```
CV accuracy : 0.961 ± 0.013
Hold-out accuracy : 0.983 (227/231)
```

Retrain whenever you add new signs or more samples — just run this script again.

---

### Step 4 — Live inference

```bash
python inference.py
```

Opens your webcam and starts predicting in real time. The predicted label appears above the bounding box around your hand. The bottom panel shows a smoothed prediction using a majority vote over the last 10 frames, which filters out single-frame noise.

Press **`Q`** to quit.

---

### Step 5 — Interactive practice UI (browser)

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

A local web app that lets you study and practise your signs interactively,
using your trained model and collected images — no extra data needed.

#### How to use it

1. **Pick a sign** — the home screen shows every label in your dataset as a
   grid of clickable tiles
2. **Study the reference images** — clicking a tile opens a practice view with
   up to 6 sample images from your `dataset/images/` folder for that sign.
   Use the thumbnail strip or the Prev / Next buttons to cycle through them
   and get a feel for how the sign looks from different angles
3. **Click "Start Camera"** — grants webcam access (one time, stays active
   while the page is open)
4. **Perform the sign** — the webcam panel runs live prediction every 350 ms.
   The predicted label and confidence appear in real time. A green **MATCH**
   badge flashes when your prediction matches the selected sign
5. **Move to the next sign** — click the back arrow to return to the grid and
   pick another label

#### Files

| File | Purpose |
|------|---------|
| `app.py` | Flask backend — serves the UI, images, and runs inference |
| `practice_ui.html` | Single-page frontend — no build step, no external dependencies |

#### Notes

- `app.py` loads `model.pkl` and `hand_landmarker.task` at startup — run
  `train_model.py` first if you have not already
- The confidence threshold in the UI is set to 55 % (lower than `inference.py`
  because the practice context is forgiving). Adjust `CONF_THRESH` at the top
  of `practice_ui.html` if needed
- The camera feed is mirrored in the browser so it feels natural, and the
  frame sent to the backend is un-mirrored before landmark detection
- The server only listens on `localhost` — nothing leaves your machine

---

## Defining your own signs

This pipeline is not limited to ASL or any particular sign language. You can teach it anything:

1. Choose a label name — it can be a letter, number, word, or emoji
2. Run `collect_data.py` and enter that label when prompted
3. Perform the gesture repeatedly and press `SPACE` to capture
4. Retrain with `train_model.py`

The only constraint is that signs must be **static** (a single held pose). Signs that require motion — like ASL J or Z — cannot be represented by a single landmark snapshot and are not supported by this pipeline.

---

## Handling unknown signs

The classifier always outputs its best guess from the labels it was trained on. If you show it a sign it has never seen, it will still predict something, possibly with low confidence. Two complementary approaches help with this:

**Confidence threshold** — inference already suppresses predictions below 65% confidence. Raise `CONF_THRESHOLD` in `inference.py` if you are seeing too many false positives.

**NONE class (recommended)** — collect 50–100 samples of random, neutral, or transitional hand poses under the label `NONE`. The model then learns what "not a valid sign" looks like and will confidently output `NONE` for unknown inputs rather than guessing. In `inference.py` you can then suppress display when the prediction is `NONE`:

```python
if raw_lbl != "NONE" and raw_conf >= CONF_THRESHOLD:
    draw_prediction(frame, raw_lbl, raw_conf, x1, y1)
```

---

## Controls

| Key | Script | Action |
|-----|--------|--------|
| `SPACE` | collect_data.py | Capture current frame |
| `N` | collect_data.py | Finish current label, enter next |
| `Q` | all scripts | Quit |

---

## Accuracy tips

| Practice | Effect |
|----------|--------|
| 50–100 samples per sign | More variation, better generalisation |
| Vary angle slightly across captures | Robust to small pose differences |
| Vary distance from camera | Landmark normalisation handles scale, but diversity still helps |
| Collect a NONE class | Clean rejection of unknown signs |
| Retrain after adding new signs | model.pkl must reflect the latest dataset |

---

## Supported static signs (ASL reference)

Letters: A B C D E F G H I K L M N O P Q R S T U V W X Y

Numbers: 0 1 2 3 4 5 6 7 8 9

> ASL letters **J** and **Z** require hand motion and are not supported. All other ASL letters and numbers work well.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Webcam capture and drawing |
| `mediapipe` | Hand landmark detection |
| `scikit-learn` | Random Forest + MLP classifier |
| `pandas` | CSV dataset management |
| `numpy` | Numerical operations |
| `joblib` | Model serialisation |