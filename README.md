# 🩺 Chest X-ray Multi-Disease Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A deep learning-based medical image classification system that automatically detects **COVID-19**, **Pneumonia**, and **Tuberculosis** from chest X-ray images using transfer learning and explainable AI.

---

## 📌 Overview

This project builds an end-to-end deep learning pipeline for multi-class disease classification from chest X-rays. Multiple CNN architectures were evaluated to identify the best-performing model while incorporating **Grad-CAM** for model interpretability.

### Key Highlights

- ✅ Multi-class disease classification
- ✅ Transfer Learning
- ✅ CNN architecture comparison
- ✅ Explainable AI using Grad-CAM
- ✅ Automated training pipeline
- ✅ Performance benchmarking

---

## 🚀 Features

- Detects:
  - COVID-19
  - Pneumonia
  - Tuberculosis

- Compared multiple deep learning architectures:
  - CNN
  - ResNet50
  - DenseNet121
  - EfficientNet-B0

- Uses:
  - Transfer Learning
  - Data Augmentation
  - Early Stopping
  - AdamW Optimizer
  - Cosine Annealing Scheduler
  - Label Smoothing
  - Grad-CAM Visualization

---

## 🏗️ Project Architecture

```text
Chest X-ray
      │
      ▼
Data Preprocessing
      │
      ▼
Data Augmentation
      │
      ▼
Deep Learning Model
(CNN / ResNet / DenseNet / EfficientNet)
      │
      ▼
Prediction
      │
      ▼
Grad-CAM Visualization
```

---

## 📂 Dataset Structure

```
dataset/
│
├── train/
│   ├── COVID19
│   ├── PNEUMONIA
│   └── TURBERCULOSIS
│
├── val/
│
└── test/
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Framework | PyTorch |
| Computer Vision | OpenCV |
| ML | Transfer Learning |
| Visualization | Grad-CAM |
| Data | NumPy, Pandas |
| Plotting | Matplotlib |

---

## ⚙️ Installation

```bash
git clone https://github.com/Pooja3706/Multi-Disease-X-ray-Detection.git

cd Multi-Disease-X-ray-Detection

pip install -r requirements.txt
```

---

## ▶️ Run Project

### Prepare Dataset

```bash
python prepare_data.py
```

### Train Models

```bash
python train_compare.py
```

### Generate Grad-CAM

```bash
python gradcam.py
```

---

## 📊 Model Comparison

| Model | Transfer Learning | Explainability |
|--------|------------------|---------------|
| CNN | ❌ | ❌ |
| ResNet50 | ✅ | ✅ |
| DenseNet121 | ✅ | ✅ |
| EfficientNet-B0 | ✅ | ✅ |

---

## 📈 Output Files

```
outputs/

├── *_best.pt
├── model_comparison.csv
├── model_comparison.json
├── classification_report.json
└── gradcam.png
```

---

## 🎯 Results

✔ DenseNet121 achieved the highest classification performance.

✔ Transfer Learning significantly improved accuracy over the custom CNN.

✔ Grad-CAM generated interpretable heatmaps highlighting clinically relevant regions.

---

## 📸 Sample Outputs

### Grad-CAM

> Add Grad-CAM image here

```
images/gradcam.png
```

### Confusion Matrix

> Add confusion matrix here

```
images/confusion_matrix.png
```

### Accuracy Plot

> Add training graph here

```
images/training_curve.png
```

---

## 📌 Future Improvements

- Vision Transformers (ViT)
- ConvNeXt
- Model Quantization
- ONNX Deployment
- FastAPI Inference API
- Streamlit Web Application

---

## 👩‍💻 Author

**Pooja Ratna Sai Sri Teku**

- GitHub: https://github.com/Pooja3706
- LinkedIn: https://linkedin.com/in/pooja-teku

---

⭐ If you found this project useful, consider giving it a star.
