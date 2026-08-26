"""
extract_audio.py

Extracts MFCC (Mel-Frequency Cepstral Coefficient) audio features from the
CREMA-D dataset and saves them as a single .npy feature matrix + labels file
that train.py can later load.

CREMA-D filenames look like: 1001_DFA_ANG_XX.wav
  1001 -> actor ID
  DFA  -> sentence code
  ANG  -> emotion code (ANG, DIS, FEA, HAP, NEU, SAD)
  XX   -> intensity level

Usage:
    python src/extract_audio.py
"""

import os
import glob
import numpy as np
import librosa
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_AUDIO_DIR = os.path.join("data", "raw", "AudioWAV")   # adjust if your CREMA-D folder differs
OUTPUT_DIR = os.path.join("data", "processed")
N_MFCC = 40           # number of MFCC coefficients to extract
SAMPLE_RATE = 16000   # resample rate for consistency across all clips

EMOTION_MAP = {
    "ANG": "Angry",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad",
}


def parse_emotion_from_filename(filepath: str) -> str | None:
    """Extract the emotion label from a CREMA-D filename."""
    filename = os.path.basename(filepath)
    parts = filename.split("_")
    if len(parts) < 3:
        return None
    emotion_code = parts[2]
    return EMOTION_MAP.get(emotion_code, None)


def extract_mfcc(filepath: str, n_mfcc: int = N_MFCC, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load an audio file and extract a fixed-length MFCC feature vector.
    We average the MFCCs across time to get one vector per clip
    (simple, robust baseline approach).
    """
    y, sr = librosa.load(filepath, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)   # shape: (n_mfcc,)
    mfcc_std = np.std(mfcc, axis=1)     # shape: (n_mfcc,)
    # Concatenate mean + std for a slightly richer fixed-length representation
    return np.concatenate([mfcc_mean, mfcc_std])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    audio_files = glob.glob(os.path.join(RAW_AUDIO_DIR, "*.wav"))
    if not audio_files:
        print(f"No .wav files found in {RAW_AUDIO_DIR}. "
              f"Make sure CREMA-D audio clips are placed there.")
        return

    print(f"Found {len(audio_files)} audio files. Extracting MFCC features...")

    features = []
    labels = []
    skipped = 0

    for filepath in tqdm(audio_files):
        emotion = parse_emotion_from_filename(filepath)
        if emotion is None:
            skipped += 1
            continue
        try:
            feat = extract_mfcc(filepath)
        except Exception as e:
            print(f"Skipping {filepath} due to error: {e}")
            skipped += 1
            continue

        features.append(feat)
        labels.append(emotion)

    features = np.array(features)
    labels = np.array(labels)

    np.save(os.path.join(OUTPUT_DIR, "audio_features.npy"), features)
    np.save(os.path.join(OUTPUT_DIR, "audio_labels.npy"), labels)

    print(f"Done. Extracted features for {len(features)} clips "
          f"({skipped} skipped). Feature shape: {features.shape}")
    print(f"Saved to {OUTPUT_DIR}/audio_features.npy and {OUTPUT_DIR}/audio_labels.npy")


if __name__ == "__main__":
    main()
