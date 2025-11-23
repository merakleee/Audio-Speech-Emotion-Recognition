# Audio Speech Emotion Recognition - Kaggle Challenge

![Score](https://img.shields.io/badge/Public%20Score-0.75498-brightgreen)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-blue)
![Python](https://img.shields.io/badge/Python-3.11-blue)

## 🎯 Overview

**Kaggle Challenge**: [Audio Speech Emotion Recognition](https://www.kaggle.com/competitions)    
**Public Score**: **0.75498** (F1-Weighted)  
**Improvement**: +5.32% over baseline (0.71178)  
**Dataset**: 1,152 training samples, 288 test samples, 8 emotion classes

---

## 🏆 Results

| Metric | Score |
|--------|-------|
| **Final Ensemble Score** | **0.75498** |
| CNN-LSTM Only | 0.710 |
| XGBoost Only | 0.667 |
| Best Weights | 40% CNN + 60% XGBoost |

---

## 🧠 Solution Architecture

```
                    AUDIO FILE
                        ↓
        ┌───────────────┴───────────────┐
        ↓                               ↓
   SEQUENTIAL FEATURES          STATISTICAL FEATURES
   (300 × 60)                   (160 dimensions)
        ↓                               ↓
   CNN-LSTM MODEL              XGBOOST MODEL
   - 2 Conv1D blocks           - 300 trees
   - LSTM layer                - Depth: 7
   - Dropout: 0.2              - Learning rate: 0.05
        ↓                               ↓
   Probability Output          Probability Output
   (8 classes)                 (8 classes)
        ↓                               ↓
        └───────────────┬───────────────┘
                        ↓
            WEIGHTED ENSEMBLE AVERAGE
            (0.4 × CNN + 0.6 × XGB)
                        ↓
                FINAL PREDICTION
                (8 emotions)
```

---

## 📊 Features Used

### CNN-LSTM Features (Sequential)
- **MFCC** (20): Spectral envelope
- **Delta** (20): Rate of change
- **Delta-Delta** (20): Acceleration of change
- Shape: 300 timesteps × 60 features

### XGBoost Features (Statistical)
- **MFCC Mean** (40)
- **MFCC Std Dev** (40)
- **MFCC Min** (40)
- **MFCC Max** (40)
- Shape: 160 dimensions

---

## 🔧 Audio Preprocessing

```
Raw Audio
    ↓
Convert to Mono (eliminate stereo)
    ↓
Normalize Volume (standardize amplitude)
    ↓
Trim Silence (remove quiet sections)
    ↓
Extract Features
```

---

## 🎓 Key Insights

1. **Ensemble Outperforms Single Models**
   - CNN-LSTM (temporal): 71.0%
   - XGBoost (statistical): 66.7%
   - **Ensemble (combined): 75.5%** ✓

2. **XGBoost > CNN-LSTM for This Task**
   - Statistical features more predictive (60% weight)
   - Shows domain-specific nature of emotion recognition

3. **Class Imbalance Solution**
   - Neutral class: 77 samples
   - Other emotions: 153-154 samples each
   - Applied class weighting to both models

4. **Weight Optimization Matters**
   - Tested 5 different weight combinations
   - (0.4, 0.6) proved optimal through Kaggle submissions

---

## 🚀 Quick Start

### Installation
```bash
pip install pandas numpy librosa scikit-learn xgboost tensorflow keras
```

### Run Pipeline
```python
# Load and preprocess audio
X_train_seq, X_train_stat = extract_features(train_files)

# Train models
cnn_model = train_cnn_lstm(X_train_seq, y_train)
xgb_model = train_xgboost(X_train_stat, y_train)

# Ensemble prediction
pred = 0.4 * cnn_model.predict(X_test_seq) + \
       0.6 * xgb_model.predict_proba(X_test_stat)
```



**Status**: ✅ Complete  
**Last Updated**: November 23, 2025

