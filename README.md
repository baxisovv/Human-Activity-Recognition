# 🏃 Human Activity Recognition — KU-HAR

> Multi-model comparison for classifying 18 physical activities from 6-axis IMU sensor data (accelerometer + gyroscope), covering classical ML, gradient boosting, Bidirectional LSTM, and a CNN-LSTM hybrid.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat)

---

## 📌 Overview

This project benchmarks a range of machine learning and deep learning approaches on the **KU-HAR dataset** — a public human activity recognition dataset collected from smartphone IMU sensors. Each sample is a 3-second window of 6-channel sensor data (300 timesteps × 6 axes), and the goal is to classify it into one of 18 activity categories.

---

## 🎯 Activity Classes (18)

| # | Activity | # | Activity |
|---|---|---|---|
| 0 | Stand | 9 | Push-up |
| 1 | Sit | 10 | Sit-up |
| 2 | Talk-sit | 11 | Walk |
| 3 | Talk-stand | 12 | Walk-backward |
| 4 | Stand-sit | 13 | Circle-walk |
| 5 | Lay | 14 | Run |
| 6 | Lay-stand | 15 | Stair-up |
| 7 | Pick | 16 | Stair-down |
| 8 | Jump | 17 | Table-tennis |

---

## 🗂️ Project Structure

```
├── ML.py                        # Full ML benchmark: 10 classifiers on hand-crafted features
├── lgbm.py                      # LightGBM with advanced feature engineering
├── 11 özellikli ML.py           # ML benchmark with a reduced 11-feature set
├── LSTM.py                      # Bidirectional LSTM on raw time-series
├── CNN ve LSTM hibrit.py        # CNN-LSTM hybrid (TimeDistributed)
├── a.py                         # Experimental variant
├── görselleştirme.py            # Standalone visualization utilities
└── Bütün modeller(ML ve DL eski hali).ipynb  # Original notebook with all models
```

**Dataset required:** `Time_domain_subsamples.csv` (KU-HAR, not included — see below)

---

## 🧠 Models

### 1. Classical ML — `ML.py`

Extracts a hand-crafted feature vector from the raw signal, then trains 10 classifiers side by side:

| Classifier | Library |
|---|---|
| Random Forest | scikit-learn |
| Extra Trees | scikit-learn |
| Gradient Boosting | scikit-learn |
| XGBoost | xgboost |
| LightGBM | lightgbm |
| CatBoost | catboost |
| SVM (RBF kernel) | scikit-learn |
| KNN (k=5) | scikit-learn |
| Logistic Regression | scikit-learn |
| Decision Tree | scikit-learn |

Results are printed in a formatted table (accuracy, precision, recall, F1, training time) and the best model gets hyperparameter tuning via `RandomizedSearchCV` with stratified 3-fold CV.

### 2. LightGBM with Advanced Features — `lgbm.py`

Focused experiment on LightGBM with richer signal features. Per channel (×6 axes), extracts:

**Time-domain:** mean, max, min, median, std, skewness, kurtosis, SMA, IQR, energy, entropy

**Frequency-domain (FFT):** dominant frequency, weighted mean frequency, spectral skewness, spectral kurtosis, spectral energy

**Autoregressive:** AR coefficients (order 4)

**Cross-channel:** pairwise Pearson correlations between Acc axes (XY, XZ, YZ) and Gyro axes (XY, XZ, YZ), plus tilt angles (X, Y, Z) from the mean acceleration vector

Total: **~129 features** per sample.

### 3. Bidirectional LSTM — `LSTM.py`

Operates directly on the raw time-series without hand-crafted features.

```
Input: (300 timesteps, 6 channels)
  → Bidirectional LSTM(64, return_sequences=True) + Dropout(0.3)
  → LSTM(64) + Dropout(0.3)
  → Dense(64, relu)
  → Dense(18, softmax)
```

Trained with `EarlyStopping` and `ReduceLROnPlateau` callbacks (up to 50 epochs, batch size 32).

### 4. CNN-LSTM Hybrid — `CNN ve LSTM hibrit.py`

Splits each 300-step window into 4 subsequences of 75 steps, applies a shared CNN to each, then passes the sequence of local features through an LSTM.

```
Input: (4 subsequences, 75 steps, 6 channels)
  → TimeDistributed Conv1D(64, kernel=3, relu)
  → TimeDistributed MaxPooling1D(2)
  → TimeDistributed Flatten
  → LSTM(128, tanh) + Dropout(0.5)
  → Dense(18, softmax)
```

Trained for 30 epochs (batch size 64, Adam lr=0.0025). The CNN captures local temporal patterns within each sub-window; the LSTM captures dependencies across them.

---

## ⚙️ Feature Engineering Pipeline

```
Raw CSV row (1800 values)
        │
        ▼
  Reshape → (6 channels, 300 timesteps)
        │
        ├── Per-channel time-domain stats (×6)
        ├── Per-channel FFT features (×6)
        ├── Per-channel AR coefficients order-4 (×6)
        └── Cross-channel correlations + tilt angles
        │
        ▼
  ~129-dimensional feature vector → StandardScaler → Classifier
```

---

## 🚀 Getting Started

### Install dependencies

```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost tensorflow matplotlib seaborn scipy
```

### Download the dataset

The project uses the **KU-HAR** dataset (Time Domain Subsamples version).

Download from: [https://data.mendeley.com/datasets/45f952y38r](https://data.mendeley.com/datasets/45f952y38r)

Place `Time_domain_subsamples.csv` in the same directory as the scripts.

### Run a model

```bash
# Classical ML benchmark (10 classifiers)
python ML.py

# LightGBM with advanced features
python lgbm.py

# Bidirectional LSTM
python LSTM.py

# CNN-LSTM hybrid
python "CNN ve LSTM hibrit.py"
```

---

## 📐 Data Format

| Field | Value |
|---|---|
| File | `Time_domain_subsamples.csv` |
| Columns | 1802 (no header) |
| Columns 0–1799 | Raw sensor values (6 channels × 300 timesteps) |
| Column 1800 | Activity class label (0–17) |
| Channels | Acc-X, Acc-Y, Acc-Z, Gyro-X, Gyro-Y, Gyro-Z |
| Sampling rate | 100 Hz |
| Window size | 3 seconds (300 samples) |

---

## 📊 Evaluation

All scripts report:
- Accuracy, Precision (macro), Recall (macro), F1 Score (macro)
- Full `classification_report` per class
- Confusion matrix heatmap
- Training/validation loss & accuracy curves (DL models)

Train/test split is **70% / 30%** with stratification across all scripts.

---

## 📄 License

MIT
