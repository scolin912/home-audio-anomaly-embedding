# Home Audio Anomaly Embedding

This project explores **audio anomaly detection** in home environments using
feature embeddings extracted from environmental sound recordings.

The goal is to build a practical pipeline that can distinguish **normal background sounds**
from **abnormal or rare events**, which can later be extended to real-time or edge AI applications.

---

## 📁 Project Structure
```
home-audio-anomaly-embedding/
├── data/
│ ├── raw/ # Original audio recordings
│ ├── wav/ # Converted / standardized WAV files
│ └── test/ # Test samples (e.g. dog bark, gunshot, scream)
├── models/ # Saved models or embedding checkpoints
├── outputs/ # Feature outputs, embeddings, or results
├── src/ # Source code for preprocessing and analysis
└── README.md

```

---

## 🔊 Audio Data

The dataset contains common **home environment sounds**, such as:

- Room tone (apartment, bedroom, kitchen)
- Human activity sounds
- Sudden or abnormal events (e.g. scream, dog bark, gunshot)

Audio files are used to study how embedding-based methods can separate
**normal patterns** from **outliers** in feature space.

---

## 🧠 Method Overview (Planned)

- Audio preprocessing (resampling, normalization)
- Feature extraction (e.g. MFCC, log-mel spectrogram)
- Embedding generation
- Anomaly detection using distance or density-based methods

This repository currently focuses on **data organization and experimentation**,
with model training and evaluation to be added incrementally.

---

## 🚀 Future Work

- Train embedding models for audio anomaly detection
- Visualize embedding space (e.g. PCA / t-SNE)
- Real-time inference experiments
- Edge AI deployment (e.g. Raspberry Pi / Jetson)

---

## 📌 Notes

This project is intended for **learning, experimentation, and portfolio demonstration**
in audio signal processing and machine learning.
