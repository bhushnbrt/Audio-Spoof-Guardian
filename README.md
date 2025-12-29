# 🕵️ Audio Deepfake Detection System (ASVspoof 5)

> **A Deep Learning system designed to detect AI-generated audio attacks (Deepfakes) using Convolutional Neural Networks (CNN) and Mel-Spectrogram analysis.**

## 📌 Project Overview
With the rise of advanced Text-to-Speech (TTS) and Voice Conversion (VC) technologies, distinguishing between real human speech and AI-generated clones has become a critical security challenge.

This project implements an end-to-end detection pipeline trained on the **ASVspoof 5** dataset. It processes raw audio into spectrograms, trains a custom CNN architecture to identify forensic artifacts, and achieves a **21.84% Equal Error Rate (EER)** on unseen development data.

## 📊 Key Results

| Metric | Score |
| :--- | :--- |
| **Validation Accuracy** | **98.32%** |
| **Validation Loss** | **0.0445** |
| **Final EER (Dev Set)** | **21.84%** |
| **Evaluated Files** | 140,950 |

*Note: The model establishes a strong baseline for a "from-scratch" architecture, significantly outperforming random probability (50%) and effectively detecting standard spoofing attacks.*

## 🛠️ Technical Architecture

### 1. Data Processing Pipeline
* **Input:** FLAC/WAV audio files (resampled to 16kHz).
* **Feature Extraction:** Mel-Spectrograms (128 bands).
* **Preprocessing:**
    * Fixed width resizing (400 time steps).
    * Silence padding for short files.
    * **Normalization:** Custom decibel scaling `(db + 80) / 80` to map values between 0.0 and 1.0.

### 2. Model Architecture (Custom CNN)
* **Framework:** TensorFlow / Keras.
* **Structure:**
    * 4x Convolutional Layers (Conv2D + ReLU).
    * MaxPolling layers for dimensionality reduction.
    * Flatten & Dense Layers (128 units).
    * **Output:** Sigmoid activation (Binary Classification: 0=Bonafide, 1=Spoof).
* **Optimization:** Adam Optimizer (`lr=0.0001`) with Binary Crossentropy loss.

## 📂 Project Structure
```bash
├── asvspoof5_epoch_04.h5      # The trained model weights (Champion Epoch)
├── train.py                   # Script to train the model from scratch
├── evaluation.py              # Script to calculate EER on the Dev set
├── demo.py                    # Live test script for single audio files
├── analyze_errors.py          # Forensic tool to find False Positives/Negatives
└── README.md                  # Project documentation
```

## 🚀 How to Run

### 1. Prerequisites
You need a Python environment with the following libraries:

```bash
conda create -n adf_env python=3.9

conda activate adf_env

pip install tensorflow pandas numpy librosa soundfile tqdm scikit-learn

# Optional: Install FFmpeg for .mp3/.m4a support
conda install -c conda-forge ffmpeg
```
### 2. Live Demo (Test a File)
To test if a specific audio file is real or fake:

1. Place your audio file (e.g., `test.wav`) in the project folder.
2. Run the demo script:

```bash
python demo.py
```
### 3. Training
To retrain the model on the ASVspoof dataset:

```bash
python train.py
```
### 4. Evaluation
To reproduce the 21.84% EER score on the full Dev set:
```bash
python evaluation.py
```

## 🔍 Forensic Analysis
Post-evaluation analysis revealed two primary failure modes:

Overconfidence (False Alarms): The model aggressively flags real audio containing microphone pops, static, or extremely short durations (<1s) as "fake."

High-Quality Attacks (Missed Detection): State-of-the-art Neural Vocoders (e.g., HiFi-GAN) that produce artifact-free spectrograms can occasionally bypass the detector, appearing "cleaner" than real human speech.

## 📜 Acknowledgments
Dataset: ASVspoof 5 Challenge (Automatic Speaker Verification Spoofing and Countermeasures).

Tools: Librosa for audio processing, TensorFlow for deep learning.
