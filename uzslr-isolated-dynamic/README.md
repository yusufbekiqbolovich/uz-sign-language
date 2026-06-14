# Uzbek Sign Language Recognition (UzSLR) — Isolated Dynamic Signs

> 🌐 Language / Til: **English** · [O'zbekcha](./README.uz.md)

<p align="center">
  <img src="docs/gifs/inference_usage.gif" alt="Inference Demo" width="600">
</p>

A complete, end-to-end machine-learning project that **recognises 50 isolated, dynamic Uzbek Sign Language (UzSL) signs** from webcam video in real time. This README explains the whole story — **how the data was collected, how the model was trained, and how to use it** — so you can both *understand* the system and *explain* it to someone else.

---

## 1. The big picture (TL;DR)

The system turns a short webcam clip of someone signing into one of 50 predicted words. It does this in five conceptual steps, each backed by a folder in this repo:

```
 Webcam video                                                     Predicted sign
      │                                                                  ▲
      ▼                                                                  │
 ┌──────────────┐   ┌─────────────┐   ┌───────────────┐   ┌──────────┐   │
 │ MediaPipe    │ → │ Select 118  │ → │ Normalise +   │ → │ CNN +    │ ──┘
 │ Holistic     │   │ landmarks   │   │ velocity/accel│   │ Transformer
 │ 543 landmarks│   │ (hands,face)│   │ → 708 features│   │ → 50 logits
 └──────────────┘   └─────────────┘   └───────────────┘   └──────────┘
```

- A model sees **32 frames** of a sign at a time. Each frame is reduced to a **708-number feature vector**, and the model outputs a probability over **50 sign classes**.
- The recognition model is a compact (~1.75 M-parameter) **hybrid 1D-CNN + Transformer**, adapted from the Kaggle ASL competition 1st-place solution (Sohn, 2023) and ported from TensorFlow to **PyTorch**.
- It reaches **~92 % validation** and **~87 % test** accuracy on the in-house Uzbek dataset.

### The full pipeline

| # | Stage | Folder | What it does | Conda env |
|---|-------|--------|--------------|-----------|
| 1 | **Video Collection** | [`video-collector/`](./video-collector/) | Record signers, extract per-frame landmarks | `video_collector_env` |
| 2 | **Dataset Preparation** | [`dataset-prep/`](./dataset-prep/) | Reorganise landmarks, split train/val/test | `video_collector_env` |
| 3 | **Data Preprocessing** | [`preprocessing/`](./preprocessing/) | Feature selection, normalisation, augmentation | `uzslr-signs` |
| 4 | **Model Training** | [`modeling/`](./modeling/) | Train + evaluate the CNN-Transformer | `uzslr-signs` |
| 5 | **Real-Time Inference** | [`inferencing/`](./inferencing/) | Desktop webcam app (OpenCV window) | `uzslr-signs` |
| 6 | **Web App** | [`web_app/`](./web_app/) | Browser app + optional LLM sentence-forming | `web-uzslr-signs` |

> Stages 1–2 build the dataset. Stages 3–4 turn it into a trained model (`best_model.pth`). Stages 5–6 are two different ways to *use* that model.
> A separate experimental branch, [`generative/`](./generative/), explores synthetic sign-video generation (DWPose + MusePose) for data augmentation — it is optional and not part of the core pipeline.

---

## 2. The 50 signs

`assalomu_alaykum` · `bahor` · `birga` · `bo'sh` · `bosh_kiyim` · `boshlanishi` · `bozor` · `eshik` · `futbol` · `iltimos` · `internet` · `javob` · `jismoniy_tarbiya` · `karam` · `kartoshka` · `kichik` · `kitob` · `ko'prik` · `likopcha` · `maktab` · `mehmonxona` · `mehribon` · `metro` · `musiqa` · `o'simlik_yog'i` · `o'ynash` · `ochish` · `ot` · `ovqat_tayyorlash` · `oxiri` · `poezd` · `pomidor` · `qidirish` · `qish` · `qo'ziqorin` · `qor` · `qorong'i` · `quyon` · `restoran` · `sariyog'` · `shokolad` · `sovun` · `stakan` · `televizor` · `tosh` · `toza` · `turish` · `yomg'ir` · `yopish` · `yordam_berish`

A visual reference (animated GIFs of every sign, two repetitions each) lives in [`show-50-signs/`](./show-50-signs/) so you can perform the signs yourself when testing the live model.

---

## 3. Setup & environments

**Python:** `3.9.23` · **Anaconda:** `conda 24.11.3`

The project deliberately uses **three isolated conda environments** (separation of concerns):

```bash
# Env A — data collection & dataset prep (stages 1–2)
cd video-collector
conda env create -f environment-video-collector.yml
conda activate video_collector_env

# Env B — preprocessing, modeling, inferencing (stages 3–5)
conda env create -f environment-uzslr-signs.yml
conda activate uzslr-signs

# Env C — web app (stage 6)
cd web_app
conda env create -f environment-web-uzslr-signs.yml
conda activate web-uzslr-signs
```

> See [`FOLDER_TREE.md`](./FOLDER_TREE.md) for where the dataset must be placed, and [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md) / [`CONDA_INFO.md`](./CONDA_INFO.md) for exact dependency hashes.

---

## 4. How the data was collected — Stage 1 ([`video-collector/`](./video-collector/))

**Goal:** capture real signers performing each sign and store both the video **and** the per-frame body landmarks.

### Who & how
- Conducted at **School No. 101, Tashkent**. **10 participants** (`signer01` … `signer10`) plus **one teacher** who clarified sign meanings. All participants were fully informed and **signed written consent forms**.
- A custom **CLI recording tool** ([`mod05_main.py`](./video-collector/mod05_main.py)) drives an OpenCV camera feed through **MediaPipe Holistic**, which detects **543 body landmarks** per frame (468 face + 33 pose + 21 + 21 hands).

### The recording workflow (3 stages)
| Stage | Action |
|-------|--------|
| **1. Signer selection** | choose / create a `signerXX` id |
| **2. Sign selection** | pick a word from a numbered list (already-recorded ones shown in green) |
| **3. Recording** | press `s` → countdown → record **exactly 32 frames** → press `s` to redo or `d` when done |

Every repetition is a fixed-length **32-frame** clip — this is why the model later always works on 32-frame windows.

### What gets stored
Each frame is saved as a NumPy file of **1662 `float32` values** (`468×3` face + `33×4` pose + `2×21×3` hands), plus the raw video:

```
video-collector/Data_Numpy_Arrays_RSL_UzSL/
└── signer{XX}/
    └── {sign}/
        ├── landmarks/rep-{XX}/frame-00.npy … frame-31.npy   ← used for training
        └── videos/rep-{XX}/video.mp4                        ← reference only
```

> The full dataset is ~3 GB and is **`.gitignored`** (not shipped in this repo). Scripts in [`video-collector/dataset-checks/`](./video-collector/dataset-checks/) validate sign counts, repetition consistency, and that every `.npy` has the exact `(1662,)` shape.

**Run it:** `cd video-collector && python mod05_main.py`

---

## 5. How the dataset was prepared — Stage 2 ([`dataset-prep/`](./dataset-prep/))

The collector's layout (grouped by signer, with videos mixed in) isn't ideal for training, so this stage:

1. **Reorganises** — copies *only* the landmark `.npy` files into a flat `data/{sign}/rep-{XX}/` folder ([`step01_reorganize_dataset.py`](./dataset-prep/step01_reorganize_dataset.py)).
2. **Splits** — divides reps into **train (80 %) / validation (10 %) / test (10 %)** ([`step02_train_val_test_split.py`](./dataset-prep/step02_train_val_test_split.py)).

```
data/
├── train/{sign}/rep-{XX}/frame-XX.npy
├── validation/{sign}/rep-{XX}/frame-XX.npy
└── test/{sign}/rep-{XX}/frame-XX.npy
```

The held-out **test set is 300 samples** (50 signs × 6 repetitions). Integrity scripts in `dataset-checks/` verify the split before and after.

> ⚠️ These scripts move/copy GB-sized data — read [`dataset-prep/README.md`](./dataset-prep/README.md) carefully first.

---

## 6. How the features are built — Stage 3 ([`preprocessing/`](./preprocessing/))

This is the heart of the system: turning raw `(32, 1662)` landmark frames into the `(32, 708)` tensor the model consumes. Implemented by the **`Preprocess`** class (see [`inferencing/inference02_preprocess.py`](./inferencing/inference02_preprocess.py) for the deployable copy).

### 6.1 Feature selection — 543 → 118 landmarks
Pose landmarks carried little useful signal, so only **118 landmarks** are kept:

| Group | Count |
|-------|-------|
| Lips | 40 |
| Left hand | 21 |
| Right hand | 21 |
| Nose | 4 |
| Right eye | 16 |
| Left eye | 16 |
| **Total** | **118** |

### 6.2 From landmarks to 708 features
For each 32-frame clip:
1. Unpack `(1662,)` → `(543, 3)` per frame.
2. Keep the **118 selected landmarks**, and only their **`x, y`** coordinates → `(T, 118, 2)`.
3. **Center & normalise** relative to a reference point (nose) and divide by the std over time.
4. Compute **velocity** (1st temporal difference) and **acceleration** (2nd difference).
5. Concatenate **position + velocity + acceleration** and flatten:

```
118 landmarks × 2 coords × 3 (pos, vel, accel) = 708 features per frame
→ final sample shape: (32, 708)
```

Velocity and acceleration are what let a *static* frame-based model understand **motion**, which is essential for *dynamic* signs.

### 6.3 Data augmentation (training only)
Applied randomly with set probabilities to make the small dataset generalise:
- **Temporal resampling** (simulate faster/slower signing)
- **Horizontal flip** with left/right landmark swapping
- **Spatial affine** (scale, rotate, shear, shift)
- **Temporal crop** to `MAX_LEN = 32`
- **Spatial masking** (rectangular landmark dropout)

> *Temporal* masking is intentionally **not** used — it would create discontinuities in the motion derivatives.

The full data flow is documented (with a PlantUML diagram) in [`preprocessing/README.md`](./preprocessing/README.md).

---

## 7. How the model was trained — Stage 4 ([`modeling/`](./modeling/))

Training and evaluation live in the notebook [`modeling/notebooks/03_ak_model_dev_v1.ipynb`](./modeling/notebooks/03_ak_model_dev_v1.ipynb).

### 7.1 Architecture — hybrid 1D-CNN + Transformer
Input `(B, 32, 708)` → logits `(B, 50)`. Defined in [`inference03_model.py`](./inferencing/inference03_model.py) (`SignLanguageModel`).

```
Input (32, 708)
   │
   ▼  Stem: Linear(708→dim) + BatchNorm
Group 1: 3 × Conv1DBlock  +  1 × TransformerBlock
Group 2: 3 × Conv1DBlock  +  1 × TransformerBlock
   │     (dim=384 large model adds Groups 3 & 4)
   ▼
Top Linear (dim → 2·dim)
   ▼
Masked Global Average Pooling over the 32 frames
   ▼
LateDropout (p=0.8) → Linear → 50 logits
```

Building blocks:
- **`Conv1DBlock`** — expand (Linear ×2) → SiLU → **causal depthwise Conv1D** (kernel 17) → BatchNorm → **ECA channel attention** → project → residual. The causal conv captures short-range temporal patterns.
- **`TransformerBlock`** — BatchNorm → **multi-head self-attention** (4 heads) → FFN (expand ×2) → residual. Captures long-range relationships across the 32 frames.
- **ECA** = Efficient Channel Attention; **LateDropout** = heavy dropout that switches on only after a warm-up.

| Config | `dim` | Groups | Params |
|--------|-------|--------|--------|
| **Base** (shipped `best_model.pth`) | 192 | 2 | ~1.75 M |
| Large | 384 | 4 | ~8–10 M |

### 7.2 Training configuration
| Setting | Value |
|---------|-------|
| Loss | CrossEntropyLoss |
| Optimizer | AdamW (`lr=5e-4`, `weight_decay=0.1`) |
| Batch size | 16 |
| Epochs | up to 300, **early stopping** (patience 15) |
| Augmentation | flip, resample, affine, spatial mask |
| Hardware / time | ~30 min on Apple MPS |

### 7.3 Results
- **~92 % validation accuracy**, **~87 % test accuracy** (300-sample test set).
- Macro-averaged **F1 ≈ 0.87** (precision 0.90 / recall 0.87).
- **Best classes** (F1 = 1.00): `maktab`, `musiqa`, `qidirish`, `qish`, `qorong'i`, `quyon`, `restoran`, `yomg'ir`, `yopish`, `o'ynash`.
- **Weakest classes:** `internet` (F1 0.29), `bozor` (0.50), and several visually similar location words (`metro`, `mehmonxona`, `poezd`) — full per-class report in [`modeling/README.md`](./modeling/README.md).

**Run training:** `cd modeling/notebooks && jupyter notebook 03_ak_model_dev_v1.ipynb`
**Outputs:** `best_model.pth` (best val accuracy) and `checkpoint.pth` (resume point).

---

## 8. How to use it (A) — Real-time desktop app ([`inferencing/`](./inferencing/))

A native OpenCV window that recognises signs live from your webcam.

```bash
conda activate uzslr-signs        # or install: pip install torch mediapipe opencv-contrib-python numpy
cd inferencing
python inference04_main.py
```

**How it behaves:**
1. Show **both hands** to the camera → status flips to `active` and a 32-frame buffer starts filling (`buffer: N/32`).
2. When the buffer hits 32, the predicted sign + confidence appear at the top, e.g. `assalomu_alaykum (0.67)`.
3. Drop your hands for >5 frames to reset; press **`q`** to quit.

**Camera note:** the capture device is set in [`inference01_config.py`](./inferencing/inference01_config.py) via `VIDEO_DEVICE` (currently `0`). If your webcam is on a different index, change it there. It auto-selects MPS / CUDA / CPU.

---

## 9. How to use it (B) — Web app ([`web_app/`](./web_app/))

A browser-based version: **MediaPipe runs in the browser**, landmarks stream to a FastAPI backend over a WebSocket, and the same PyTorch model predicts. It also has an optional **LLM mode** that strings recognised signs into a grammatical Uzbek sentence using a local Ollama model.

```bash
# Easiest — Docker (sign recognition only)
docker pull 00015775/uzslr-web:1.0.0
docker run -p 7860:7860 00015775/uzslr-web:1.0.0      # → http://localhost:7860

# Local Python (sign recognition only)
conda activate web-uzslr-signs
uvicorn web_app.backend.main:app --port 8000          # → http://localhost:8000
```

For the LLM sentence-forming feature (`:2.0.0` image or `LLM_ENABLED=true`), and full local/Docker instructions, see [`web_app/README.md`](./web_app/README.md). Docker images: <https://hub.docker.com/repository/docker/00015775/uzslr-web>.

---

## 10. Repository map

```
uzslr-isolated-dynamic/
├── video-collector/   # Stage 1 — record signs + landmarks (CLI + MediaPipe)
├── dataset-prep/      # Stage 2 — reorganise + train/val/test split
├── preprocessing/     # Stage 3 — feature engineering notebooks
├── modeling/          # Stage 4 — training notebook + best_model.pth, checkpoint.pth
├── inferencing/       # Stage 5 — real-time desktop webcam app
├── web_app/           # Stage 6 — FastAPI + browser app (+ optional LLM)
├── generative/        # (optional) synthetic sign-video generation (DWPose/MusePose)
├── show-50-signs/     # reference GIFs for all 50 signs
├── docs/              # images & GIFs used in documentation
└── *.md / *.yml       # top-level docs and conda environment files
```

---

## 11. Acknowledgements & credits

The **preprocessing and modeling pipelines are logical adaptations** of the Kaggle "Isolated Sign Language Recognition" 1st-place solution by **Sohn, H. (2023)** (the *Hoyso48* notebook). The original was written in **TensorFlow** for ASL; here it is **re-implemented in PyTorch** and adapted to a low-resource **Uzbek** dataset with a different data structure — it is **not a direct copy**.
Reference: [Sohn, H., 2023 – 1st place solution (1D-CNN + Transformer)](https://www.kaggle.com/code/hoyso48/1st-place-solution-training).

Sincere thanks to **School No. 101, Tashkent** — the 10 participants and teacher who contributed sign translations, meanings, and context. All participants gave **written informed consent**; for privacy, the people are not shown in the `show-50-signs/` previews (those were re-performed by the author).

<details>
<summary><b>Further references</b></summary>

- Hoyso48 (2023). *1st place solution – training* [Kaggle notebook].
- Bergeron, M. (2024). *Insightful Datasets for ASL recognition*. Hackster.io.
- Cookiecutter Data Science — project/naming template.

</details>
