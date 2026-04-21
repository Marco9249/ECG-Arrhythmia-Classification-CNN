<div align="center">

<img src="https://img.shields.io/badge/%E2%9D%A4%EF%B8%8F-Clinical%20AI-E63946?style=for-the-badge&labelColor=1d3557" alt="Clinical AI"/>

# 1D-CNN Arrhythmia Classification System

### 🫀 *High-Fidelity Electrocardiogram (ECG) Classification via Deep Learning* 🫀

<br/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/SMOTE-Imbalanced--Learn-8B5CF6?style=for-the-badge)](https://imbalanced-learn.org/)
[![Dataset](https://img.shields.io/badge/Data-MIT--BIH%20Arrhythmia-005288?style=for-the-badge)](https://www.physionet.org/content/mitdb/1.0.0/)

<br/>

<img src="https://img.shields.io/badge/Author-Mohammed%20Ezzeldin%20Babiker%20Abdullah-4A90D9?style=flat-square&logo=google-scholar&logoColor=white" alt="Author"/>

---

*"Enhancing clinical diagnostics with explainable, robust, and highly accurate one-dimensional convolutional neural networks."*

</div>

> [!IMPORTANT]
> **Implementation Note**: This repository contains the core architecture and settings as described in the associated research paper. However, some code structures and experimental configurations have been slightly adjusted to facilitate educational study, modification, and independent testing. The codebase will be fully synchronized with the exact methodology presented in the manuscript upon the paper's final formal publication.

---

## 🎯 Project Overview

This repository contains the complete implementation for a highly optimized 1D Convolutional Neural Network (1D-CNN) designed for **multi-class ECG arrhythmia classification** using the globally recognized **MIT-BIH Arrhythmia Dataset**. 

The system leverages Synthetic Minority Over-sampling Technique (SMOTE) to overcome severe clinical class imbalance and incorporates Grad-CAM visual interpretability to provide clinical trust in the model's decisions.

### ✨ Key Contributions

| Contribution | Description |
|:------------:|:------------|
| ⚖️ **SMOTE Balancing** | Solves extreme MIT-BIH class imbalance (Group N dominates 82% of data) |
| 🧬 **1D Convolutional Layers** | Tailored to extract morphological wave structures from 187-timestep ECG signals |
| 🔍 **Explainable AI (XAI)** | 1D Grad-CAM implementation maps exactly *which* part of the QRS complex triggered the diagnosis |
| 📊 **Research-Grade Visuals** | Generates 10 fully automated, 4K true-white background clinical charts (Loss, Precision, Heatmaps) |

---

## 🫀 Arrhythmia Classes Modeled

The network successfully categorizes 187-timestep waveforms into the AAMI recognized super-classes:
1. **N (Normal)** - Normal beats & structural variations
2. **S (Supraventricular)** - Premature or ectopic beats
3. **V (Ventricular)** - Ventricular premature contraction
4. **F (Fusion)** - Fusion of ventricular and normal beats
5. **Q (Unknown)** - Paced beats, unclassified

---

## 🏗️ Architecture Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   🫀  ECG Signal Vector (Length: 187)                        │
│                                                             │
│  ┌───────────────────────▼───────────────────────┐          │
│  │  Block 1: Conv1D (64 filters)                 │ High-    │
│  │  Batch Norm → ReLU → Max Pooling (Pool=2)     │ Freq     │
│  └───────────────────────┬───────────────────────┘          │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────┐          │
│  │  Block 2: Conv1D (128 filters)                │ Wave     │
│  │  Batch Norm → ReLU → Max Pooling (Pool=2)     │ Capture  │
│  └───────────────────────┬───────────────────────┘          │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────┐          │
│  │  Block 3: Conv1D (256 filters)                │ Deep     │
│  │  Batch Norm → ReLU                            │ Morph.   │
│  └───────────────────────┬───────────────────────┘          │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────┐          │
│  │  Global Average Pooling 1D                    │          │
│  │  Flattening structural sequence               │          │
│  └───────────────────────┬───────────────────────┘          │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────┐          │
│  │  Fully Connected Integrator                   │          │
│  │  Dense (128) + Dropout (0.3)                  │          │
│  └───────────────────────┬───────────────────────┘          │
│                          │                                  │
│        📋 Softmax Classifier (5 Probabilities)              │
│        [ N, S, V, F, Q ]                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
📦 ECG-Arrhythmia-Classification-CNN/
│
├── 📁 training_code/
│   └── 🧠 ecg_cnn_classifier.py         # Full Model, SMOTE, and Metrics Pipeline
│
├── 📁 training_data/
│   ├── 📊 mitbih_train.csv              # MIT-BIH Training Matrix
│   └── 📊 mitbih_test.csv               # MIT-BIH Testing Matrix
│
├── 📄 ECG_Arrhythmia_Classification_Paper.docx # Academic Manuscript
├── 📋 requirements.txt
└── 📖 README.md
```

---

## 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute Full Clinical Pipeline:**
   ```bash
   python training_code/ecg_cnn_classifier.py
   ```

3. **Outputs Generated Automatically:**
   - Saved `Final_Clinical_Model.keras` engine
   - Sub-directory `ecg_outputs/` populated with 10 Research-Grade Visuals (Data distribution, Confusion matrices, F1-Score mappings, and Grad-CAM interpretability heatmaps)

---

## 📚 Related Research Portfolio

<div align="center">

| # | Paper | Repository |
|:-:|:------|:----------:|
| 1 | Physics-Guided CNN-BiLSTM Solar Forecast | [![Repo](https://img.shields.io/badge/Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/Physics-Guided-CNN-BiLSTM-Solar) |
| 2 | Physics-Informed State Space Model (PI-SSM) | [![Repo](https://img.shields.io/badge/Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/PI-SSM-Solar-Forecasting) |
| 3 | PI-SSM Cross-Attention Networks | [![Repo](https://img.shields.io/badge/Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/PI-SSM-CrossAttention-Solar) |
| ... | ... | ... |
| 6 | DeepAR Probabilistic Forecasting | [![Repo](https://img.shields.io/badge/Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/DeepAR-Probabilistic-Load-Forecasting) |
| **7** | **ECG Arrhythmia Classification** *(this repo)* 🌟 | [![Repo](https://img.shields.io/badge/Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/ECG-Arrhythmia-Classification-CNN) |

</div>

---

## 📖 Citation

```bibtex
@misc{abdullah2026ecgcnn,
  title   = {Automated Cardiac Arrhythmia Classification using Deep 1D Convolutional Neural Networks},
  author  = {Mohammed Ezzeldin Babiker Abdullah},
  year    = {2026}
}
```

---

<div align="center">

### 👤 Author

**Mohammed Ezzeldin Babiker Abdullah**
*Researcher in Predictive Modeling & Deep Learning*

[![GitHub](https://img.shields.io/badge/GitHub-Marco9249-181717?style=for-the-badge&logo=github)](https://github.com/Marco9249)

</div>

