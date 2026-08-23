# Emotion Recognition Baseline

Reproducing a baseline emotion recognition result on CREMA-D using audio (MFCC)
and facial (MediaPipe landmark) features, with early fusion.

## Setup

1. Clone this repo
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # on Mac/Linux
   venv\Scripts\activate         # on Windows

   pip install -r requirements.txt
   ```
3. Download CREMA-D and place it into `data/raw/`
4. Run the scripts in `src/` in order:
   ```bash
   python src/extract_audio.py
   python src/extract_video.py
   python src/train.py
   python src/evaluate.py
   ```

## Project Structure

```
emotion-recognition-baseline/
│
├── data/
│   ├── raw/          # original downloaded dataset (not tracked in git)
│   └── processed/    # extracted features (audio/video)
│
├── src/
│   ├── extract_audio.py   # MFCC feature extraction
│   ├── extract_video.py   # MediaPipe facial landmark extraction
│   ├── train.py            # trains audio-only, video-only, and fusion models
│   └── evaluate.py         # reports accuracy, F1, confusion matrix
│
├── results/           # saved metrics, plots, confusion matrices
├── requirements.txt
├── README.md
└── .gitignore
```

## Results

_(to be filled in after training)_

| Model            | Accuracy | F1-score |
|------------------|----------|----------|
| Audio only       | TBD      | TBD      |
| Facial only      | TBD      | TBD      |
| Early fusion     | TBD      | TBD      |

## Notes

- Dataset: [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
- Emotion classes: Angry, Disgust, Fear, Happy, Neutral, Sad
