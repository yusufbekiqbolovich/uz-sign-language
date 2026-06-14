"""
train_model.py — Train sign language classifier
================================================
Reads dataset/landmarks.csv, trains a Random Forest + MLP ensemble,
evaluates with cross-validation, and saves the model to model.pkl.

Usage:
    python train_model.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble         import RandomForestClassifier, VotingClassifier
from sklearn.neural_network   import MLPClassifier
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.model_selection  import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics          import classification_report, confusion_matrix
from sklearn.pipeline         import Pipeline

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE_DIR, "dataset", "landmarks.csv")
MODEL_OUT = os.path.join(BASE_DIR, "model.pkl")

# ── Feature columns (63 floats: x0..z20) ─────────────────────────────────────
FEATURE_COLS = [f"{ax}{i}" for i in range(21) for ax in ("x","y","z")]

def load_data():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: Dataset not found at {CSV_PATH}")
        print("Run collect_data.py first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples from {CSV_PATH}")

    # drop rows with missing landmark values
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with missing landmarks.")

    return df

def check_minimum_samples(df):
    counts = df["label"].value_counts()
    print("\nSamples per label:")
    for lbl, cnt in counts.items():
        bar    = "█" * cnt
        status = "" if cnt >= 10 else "  ⚠ low — aim for 30+"
        print(f"  {lbl:>4} : {cnt:>3}  {bar}{status}")

    low = counts[counts < 5]
    if len(low) > 0:
        print(f"\nWARNING: labels {list(low.index)} have <5 samples. "
              "Collect more data for reliable accuracy.")
        if len(low) == len(counts):
            print("FATAL: All labels have <5 samples. Cannot train.")
            sys.exit(1)

def build_model():
    """
    Soft-voting ensemble of Random Forest + MLP.
    Both wrapped in a pipeline with standard scaling.
    RF doesn't need scaling but it doesn't hurt either.
    """
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )),
    ])

    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )),
    ])

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("mlp", mlp)],
        voting="soft",
    )
    return ensemble

def main():
    print("=" * 55)
    print("  Sign Language Model Trainer")
    print("=" * 55)

    df = load_data()
    check_minimum_samples(df)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y_raw = df["label"].values

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    n_classes = len(le.classes_)
    print(f"\nClasses ({n_classes}): {list(le.classes_)}")
    print(f"Total samples : {len(X)}")

    # ── Cross-validation ──────────────────────────────────────────────────────
    n_splits = min(5, df["label"].value_counts().min())  # cap splits by rarest class
    n_splits = max(2, n_splits)
    print(f"\nRunning {n_splits}-fold stratified cross-validation …")

    model = build_model()
    cv    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=1)

    print(f"  CV accuracy : {scores.mean():.3f} ± {scores.std():.3f}")
    for fold, s in enumerate(scores, 1):
        bar = "█" * int(s * 30)
        print(f"  Fold {fold}     : {s:.3f}  {bar}")

    # ── Hold-out evaluation ───────────────────────────────────────────────────
    test_size = min(0.2, max(0.1, 1 - (10 * n_classes / len(X))))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    print(f"\nFitting final model on {len(X_train)} samples …")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = (y_pred == y_test).mean()
    print(f"Hold-out accuracy : {acc:.3f} ({int(acc * len(y_test))}/{len(y_test)})")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_]))

    # ── Confusion matrix (text) ───────────────────────────────────────────────
    cm     = confusion_matrix(y_test, y_pred)
    labels = [str(c) for c in le.classes_]
    col_w  = max(len(l) for l in labels) + 2
    header = " " * col_w + "".join(f"{l:>{col_w}}" for l in labels)
    print("Confusion matrix (rows=true, cols=pred):")
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{labels[i]:>{col_w}}" + "".join(f"{v:>{col_w}}" for v in row)
        print(row_str)

    # ── Retrain on full dataset & save ────────────────────────────────────────
    print(f"\nRetraining on full dataset ({len(X)} samples) …")
    model.fit(X, y)

    payload = {"model": model, "label_encoder": le, "feature_cols": FEATURE_COLS}
    joblib.dump(payload, MODEL_OUT)
    print(f"\n✓ Model saved → {MODEL_OUT}")
    print(f"✓ Labels      : {list(le.classes_)}")

if __name__ == "__main__":
    main()
