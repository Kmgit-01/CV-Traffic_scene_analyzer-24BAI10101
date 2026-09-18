# 🚦 Traffic Scene Analyzer

**Teach a computer to read a stop sign and follow a car — without a single neural network.**

A command-line computer vision pipeline that classifies traffic sign images and detects/tracks moving vehicles in video, built entirely on classical image processing and machine learning: enhancement, edge & corner detection, the Hough transform, segmentation & clustering, PCA, Naive Bayes / KNN, and object detection with tracking.

No GPUs. No deep learning frameworks. No GUI. Just a terminal, OpenCV, and the math behind it.

---

## Table of Contents

- [Why this project](#why-this-project)
- [How it works](#how-it-works)
- [Features](#features)
- [Technologies used](#technologies-used)
- [Project structure](#project-structure)
- [Setup instructions](#setup-instructions)
- [Running the project](#running-the-project)
- [Testing](#testing)
- [Notes](#notes)

---

## Why this project

Two everyday computer-vision problems, one pipeline:

1. **"What sign is that?"** — clean up a photo, pull out its shape and color features, and classify it.
2. **"Which car is which?"** — watch a video and keep a consistent ID on every vehicle as it moves.

Rather than solving these as disconnected homework exercises, this project chains classical techniques together into a single working system, so the output of one stage genuinely feeds the next.

## How it works

```
 Image / Video Input
        │
        ▼
 ┌─────────────────────┐
 │ 1. Preprocessing     │  noise removal · log/power-law transforms
 │    & Enhancement     │  morphology · histogram equalization
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐
 │ 2. Feature Extraction│  Canny edges · Harris & Shi-Tomasi corners
 │    & Segmentation    │  Hough transform · K-means / K-medoids · PCA
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐
 │ 3. Classification &  │  KNN vs. Naive Bayes · background-subtraction
 │    Tracking          │  detection · centroid-based multi-object tracking
 └─────────┬───────────┘
           ▼
   Predicted label / annotated video
```

## Features

- 🧼 **Enhance** an image — denoise, gamma-correct, equalize histogram
- 🔁 **Represent** an image — flip and/or reduce contrast
- 🧩 **Segment** an image with K-means and watershed
- 🎯 **Train** KNN and Naive Bayes classifiers on a labeled dataset, automatically keeping whichever performs better
- 🏷️ **Classify** a new sign image with the trained model
- 🚗 **Detect & track** moving objects across a video, saving an annotated output with per-object IDs
- 📝 Every stage is timed and logged to `outputs/run.log` — nothing runs silently

## Technologies used

| Tool | Purpose |
|---|---|
| Python 3.9+ | Core language |
| OpenCV (`opencv-python-headless`) | All image/video processing |
| NumPy | Array math throughout |
| scikit-learn | PCA, KNN, Gaussian Naive Bayes, train/test split |
| pytest | Unit tests |

## Project structure

```
traffic-scene-analyzer/
├── main.py                  # CLI entry point
├── preprocessing.py         # Module 1: enhancement, flip, contrast
├── feature_extraction.py    # Module 2: edge/corner/Hough detection
├── segmentation.py          # Module 2: clustering + PCA
├── classifier.py            # Module 3: KNN + Naive Bayes
├── detection_tracking.py    # Module 3: detection + centroid tracker
├── train_classifier.py      # Offline training script
├── utils.py                 # Logging, validation helpers
├── tests/test_pipeline.py   # Unit tests
├── data/train/<class>/      # Sample labeled training images
├── models/                  # Saved trained classifier (generated)
├── outputs/                 # Generated results and run.log
└── requirements.txt
```

## Setup instructions

These steps assume the evaluator's machine has nothing pre-installed and no prior context about this project.

### 1. Prerequisites

- Python 3.9 or later, available on the command line
  ```bash
  python --version   # or: python3 --version
  ```
- `pip` available
  ```bash
  pip --version
  ```

### 2. Clone the repository

```bash
git clone https://github.com/<github-username>/traffic-scene-analyzer.git
cd traffic-scene-analyzer
```

### 3. Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configuration

There's nothing to configure — no API keys, no `.env` file, no external services. Every path is a plain command-line argument, and the default paths already point at the `data/`, `models/`, and `outputs/` folders shipped in this repo.

## Running the project

Run everything from the project root, with the virtual environment activated. Every feature is a `main.py` subcommand — no GUI is ever launched.

### Train the classifier

A small sample dataset (`data/train/circle`, `data/train/square`, `data/train/triangle`) ships with the repo so the pipeline runs out of the box. To use a real dataset such as GTSRB instead, replace the contents of `data/train/` with one folder per class, containing that class's images.

```bash
python train_classifier.py --data data/train --pca-components 8 --k 3
```

This trains both KNN and Naive Bayes, prints their accuracy, keeps whichever scores higher, and saves it to `models/classifier.pkl`.

### Classify a sign image

```bash
python main.py classify --img data/train/circle/c0.png
```

### Enhance an image

```bash
python main.py enhance --img data/train/circle/c0.png --out outputs/enhanced.png
```

### Flip / reduce contrast of an image

```bash
python main.py represent --img data/train/circle/c0.png --flip horizontal --contrast 0.6 --out outputs/represented.png
```

`--flip` accepts `horizontal`, `vertical`, or `both`. `--contrast` is a factor in `[0, 1]` (1.0 = unchanged, 0.0 = flat gray). Both flags are optional and independent.

### Segment an image

```bash
python main.py segment --img data/train/circle/c0.png --k 3
```

Writes `outputs/segmentation_kmeans.png` and `outputs/segmentation_watershed.png`.

### Detect and track objects in a video

```bash
python main.py track --video path/to/your_clip.mp4 --out outputs/tracked_output.mp4
```

Every detected object gets a bounding box and a stable ID that persists across frames.

## Testing

Unit tests cover preprocessing, feature extraction, segmentation, and the centroid tracker.

```bash
python -m pytest tests/ -v
```

## Notes

- Every run appends to `outputs/run.log`, including per-stage timing — a handy trail if anything looks off.
- The bundled sample dataset is synthetic (drawn shapes), included purely so the pipeline is runnable without first downloading an external dataset. See `statement.md` and the project report for the intended real-world dataset (GTSRB) and the reasoning behind it.
