"""
train.py

Trains three baseline emotion classifiers using the features extracted by
extract_audio.py and extract_video.py:
  1. Audio-only  (MFCC features)
  2. Video-only  (MediaPipe facial landmark features)
  3. Early fusion (audio + video features concatenated)

Uses a simple MLP (multi-layer perceptron) classifier via scikit-learn as
the baseline model -- intentionally simple, since the goal is a reproducible
baseline, not a state-of-the-art result.

Saves trained models and basic metrics to results/.

Usage:
    python src/train.py
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
import joblib

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROCESSED_DIR = os.path.join("data", "processed")
RESULTS_DIR = "results"
RANDOM_STATE = 42
TEST_SIZE = 0.15

MLP_PARAMS = dict(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    max_iter=500,
    random_state=RANDOM_STATE,
    early_stopping=True,
)


def load_features(name: str):
    """Load a features/labels pair saved by the extraction scripts."""
    feat_path = os.path.join(PROCESSED_DIR, f"{name}_features.npy")
    label_path = os.path.join(PROCESSED_DIR, f"{name}_labels.npy")

    if not os.path.exists(feat_path) or not os.path.exists(label_path):
        return None, None

    features = np.load(feat_path)
    labels = np.load(label_path, allow_pickle=True)
    return features, labels


def train_and_evaluate(X, y, model_name: str):
    """Train an MLP classifier and return the fitted model/scaler/encoder + metrics."""
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = MLPClassifier(**MLP_PARAMS)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "uar": float(recall_score(y_test, y_pred, average="macro")),  # UAR = macro recall
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classes": label_encoder.classes_.tolist(),
    }

    print(f"\n[{model_name}] Results:")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  UAR       : {metrics['uar']:.4f}")

    return clf, scaler, label_encoder, metrics


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_metrics = {}

    # -------------------------------------------------------------
    # Audio-only
    # -------------------------------------------------------------
    audio_features, audio_labels = load_features("audio")
    if audio_features is not None:
        clf, scaler, encoder, metrics = train_and_evaluate(
            audio_features, audio_labels, "Audio-only"
        )
        joblib.dump(clf, os.path.join(RESULTS_DIR, "audio_model.joblib"))
        joblib.dump(scaler, os.path.join(RESULTS_DIR, "audio_scaler.joblib"))
        all_metrics["audio_only"] = metrics
    else:
        print("Audio features not found -- run extract_audio.py first. Skipping audio-only model.")

    # -------------------------------------------------------------
    # Video-only
    # -------------------------------------------------------------
    video_features, video_labels = load_features("video")
    if video_features is not None:
        clf, scaler, encoder, metrics = train_and_evaluate(
            video_features, video_labels, "Video-only"
        )
        joblib.dump(clf, os.path.join(RESULTS_DIR, "video_model.joblib"))
        joblib.dump(scaler, os.path.join(RESULTS_DIR, "video_scaler.joblib"))
        all_metrics["video_only"] = metrics
    else:
        print("Video features not found -- run extract_video.py first. Skipping video-only model.")

    # -------------------------------------------------------------
    # Early fusion (audio + video concatenated)
    # -------------------------------------------------------------
    # NOTE: This simplified baseline assumes extract_audio.py and
    # extract_video.py produce label arrays that line up 1:1 by index
    # (i.e. same clip order). For a more robust pipeline, extend both
    # extraction scripts to save a shared clip ID per row and join on
    # that ID here instead of relying on index alignment.
    if audio_features is not None and video_features is not None:
        n = min(len(audio_labels), len(video_labels))

        if list(audio_labels[:n]) == list(video_labels[:n]):
            fused_features = np.concatenate(
                [audio_features[:n], video_features[:n]], axis=1
            )
            fused_labels = audio_labels[:n]

            clf, scaler, encoder, metrics = train_and_evaluate(
                fused_features, fused_labels, "Early fusion (audio + video)"
            )
            joblib.dump(clf, os.path.join(RESULTS_DIR, "fusion_model.joblib"))
            joblib.dump(scaler, os.path.join(RESULTS_DIR, "fusion_scaler.joblib"))
            all_metrics["early_fusion"] = metrics
        else:
            print(
                "\nWarning: audio and video labels do not line up 1:1 by index. "
                "Fusion requires per-clip aligned features -- see the note above "
                "match_by_common_ids(). Skipping fusion model for now."
            )
    else:
        print("Skipping fusion model -- need both audio and video features first.")

    # -------------------------------------------------------------
    # Save summary metrics
    # -------------------------------------------------------------
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nAll done. Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
