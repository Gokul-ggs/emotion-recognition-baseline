"""
extract_video.py

Extracts facial landmark features from CREMA-D video clips using MediaPipe
Face Mesh, and saves them as a single .npy feature matrix + labels file
that train.py can later load.

For each video, we sample frames at a fixed interval, run MediaPipe Face
Mesh on each sampled frame, and average the landmark positions across
frames to get one fixed-length feature vector per clip.

CREMA-D filenames look like: 1001_DFA_ANG_XX.flv
  1001 -> actor ID
  DFA  -> sentence code
  ANG  -> emotion code (ANG, DIS, FEA, HAP, NEU, SAD)
  XX   -> intensity level

Usage:
    python src/extract_video.py
"""

import os
import glob
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_VIDEO_DIR = os.path.join("data", "raw", "VideoFlash")  # adjust if your CREMA-D folder differs
OUTPUT_DIR = os.path.join("data", "processed")
FRAME_SAMPLE_INTERVAL = 5   # process every Nth frame (speed vs. accuracy trade-off)
NUM_LANDMARKS = 468         # MediaPipe Face Mesh landmark count

EMOTION_MAP = {
    "ANG": "Angry",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad",
}

mp_face_mesh = mp.solutions.face_mesh


def parse_emotion_from_filename(filepath: str) -> str | None:
    """Extract the emotion label from a CREMA-D filename."""
    filename = os.path.basename(filepath)
    parts = filename.split("_")
    if len(parts) < 3:
        return None
    emotion_code = parts[2]
    return EMOTION_MAP.get(emotion_code, None)


def extract_landmarks_from_video(filepath: str, face_mesh) -> np.ndarray | None:
    """
    Sample frames from a video, run MediaPipe Face Mesh on each, and return
    the mean (x, y, z) landmark positions across all frames where a face
    was detected. Returns None if no face was ever detected.
    """
    cap = cv2.VideoCapture(filepath)
    frame_landmarks = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SAMPLE_INTERVAL == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
                )  # shape: (468, 3)
                frame_landmarks.append(coords.flatten())  # shape: (1404,)

        frame_idx += 1

    cap.release()

    if not frame_landmarks:
        return None

    frame_landmarks = np.array(frame_landmarks)
    return np.mean(frame_landmarks, axis=0)  # shape: (1404,)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_files = glob.glob(os.path.join(RAW_VIDEO_DIR, "*.flv"))
    if not video_files:
        # Some CREMA-D mirrors use .mp4 instead of .flv
        video_files = glob.glob(os.path.join(RAW_VIDEO_DIR, "*.mp4"))

    if not video_files:
        print(f"No video files found in {RAW_VIDEO_DIR}. "
              f"Make sure CREMA-D video clips are placed there.")
        return

    print(f"Found {len(video_files)} video files. Extracting facial landmarks...")

    features = []
    labels = []
    skipped = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:

        for filepath in tqdm(video_files):
            emotion = parse_emotion_from_filename(filepath)
            if emotion is None:
                skipped += 1
                continue

            try:
                feat = extract_landmarks_from_video(filepath, face_mesh)
            except Exception as e:
                print(f"Skipping {filepath} due to error: {e}")
                skipped += 1
                continue

            if feat is None:
                skipped += 1
                continue

            features.append(feat)
            labels.append(emotion)

    features = np.array(features)
    labels = np.array(labels)

    np.save(os.path.join(OUTPUT_DIR, "video_features.npy"), features)
    np.save(os.path.join(OUTPUT_DIR, "video_labels.npy"), labels)

    print(f"Done. Extracted features for {len(features)} clips "
          f"({skipped} skipped). Feature shape: {features.shape}")
    print(f"Saved to {OUTPUT_DIR}/video_features.npy and {OUTPUT_DIR}/video_labels.npy")


if __name__ == "__main__":
    main()
