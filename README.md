# 🩺 Chest X-Ray Disease Classification using EfficientNet-B0

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![EfficientNet](https://img.shields.io/badge/Model-EfficientNet--B0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-FF4B4B)
![Grad-CAM](https://img.shields.io/badge/XAI-Grad--CAM-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

A deep learning-based chest X-ray classification system that uses **EfficientNet-B0** with transfer learning to classify X-ray images into **COVID-19, Pneumonia, or Tuberculosis**. The project also uses **Grad-CAM** to visualize the regions that influenced the model's prediction and provides an interactive **Streamlit web application** for inference.

---

## 📌 Overview

This project demonstrates an end-to-end deep learning pipeline for classifying chest X-ray images into three disease categories:

- 🦠 COVID-19
- 🫁 Pneumonia
- 🩻 Tuberculosis

The system uses **EfficientNet-B0**, a pretrained convolutional neural network, through transfer learning. The pretrained model is adapted to classify the three target classes.

To improve interpretability, **Grad-CAM (Gradient-weighted Class Activation Mapping)** is used to generate a heatmap highlighting the regions that contributed to the model's prediction.

The project also includes a **Streamlit web application** where users can upload a chest X-ray and view the predicted class, confidence score, class probabilities, and Grad-CAM visualization.

---

## ✨ Key Highlights

- ✅ Three-class chest X-ray classification
- ✅ EfficientNet-B0 transfer learning
- ✅ Image preprocessing and augmentation
- ✅ Early stopping
- ✅ AdamW optimizer
- ✅ Cosine Annealing learning-rate scheduler
- ✅ Label smoothing
- ✅ Grad-CAM explainability
- ✅ Prediction confidence and probabilities
- ✅ Interactive Streamlit web application
- ✅ GPU-supported training using PyTorch

---

## 🚀 Features

### 🔍 Disease Classification

The trained model classifies a chest X-ray into one of the following classes:

```text
COVID-19
Pneumonia
Tuberculosis
```

### 🧠 Transfer Learning

The project uses **EfficientNet-B0 pretrained on ImageNet**.

The final classification layer is modified to predict the three target classes.

### 🔥 Grad-CAM Visualization

Grad-CAM is used to generate a visual explanation of the model's prediction.

The generated heatmap highlights image regions that contributed to the predicted class.

### 🌐 Streamlit Web Application

The Streamlit application allows users to:

1. Upload a chest X-ray image
2. Run the trained EfficientNet-B0 model
3. View the predicted disease
4. View prediction confidence
5. View class probabilities
6. View Grad-CAM visualization

---

## 🏗️ Project Architecture

```text
                    Chest X-Ray Image
                           │
                           ▼
                  Image Preprocessing
                           │
                           ▼
                    Data Augmentation
                           │
                           ▼
                    EfficientNet-B0
                  Transfer Learning
                           │
                           ▼
                     Classification
                           │
                           ▼
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          COVID-19      Pneumonia    Tuberculosis
                           │
                           ▼
                        Grad-CAM
                           │
                           ▼
                  Explainable Heatmap
                           │
                           ▼
                   Streamlit Web App
```

---

## 📂 Project Structure

```text
Chest-XRay-Disease-Classification/
│
├── dataset/
│   ├── train/
│   │   ├── COVID19/
│   │   ├── PNEUMONIA/
│   │   └── TURBERCULOSIS/
│   │
│   ├── val/
│   │   ├── COVID19/
│   │   ├── PNEUMONIA/
│   │   └── TURBERCULOSIS/
│   │
│   └── test/
│       ├── COVID19/
│       ├── PNEUMONIA/
│       └── TURBERCULOSIS/
│
├── outputs/
│   ├── efficientnet_b0_best.pt
│   ├── efficientnet_b0_classification_report.json
│   ├── model_comparison.csv
│   ├── model_comparison.json
│   └── gradcam_test.png
│
├── app.py
├── gradcam.py
├── prepare_data.py
├── train_compare.py
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** The dataset and trained model checkpoint are excluded from GitHub because of their large file sizes.

---

## 📊 Dataset

The project uses a chest X-ray dataset containing images belonging to three classes:

- COVID-19
- Pneumonia
- Tuberculosis

The dataset is organized into training, validation, and testing splits.

```text
dataset/
│
├── train/
├── val/
└── test/
```

Each split contains the three target classes.

### Dataset Source

The dataset used in this project is:

**Chest X-Ray (Pneumonia, Covid-19, Tuberculosis)**

The dataset is used for educational and research purposes.

---

## 🧹 Data Preparation

The `prepare_data.py` script prepares the required three-class dataset structure.

Run:

```bash
python prepare_data.py
```

The resulting dataset structure is:

```text
dataset/
│
├── train/
│   ├── COVID19/
│   ├── PNEUMONIA/
│   └── TURBERCULOSIS/
│
├── val/
│   ├── COVID19/
│   ├── PNEUMONIA/
│   └── TURBERCULOSIS/
│
└── test/
    ├── COVID19/
    ├── PNEUMONIA/
    └── TURBERCULOSIS/
```

---

## 🧠 Model

### EfficientNet-B0

This project uses **EfficientNet-B0** as the final deep learning model.

The model is initialized with pretrained ImageNet weights and fine-tuned for the three-class chest X-ray classification task.

```text
                    Input Image
                         │
                         ▼
                  EfficientNet-B0
                         │
                         ▼
                  Feature Extraction
                         │
                         ▼
                 Classification Layer
                         │
                         ▼
                    3-Class Output
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
         COVID-19    Pneumonia   Tuberculosis
```

### Training Configuration

| Parameter | Value |
|---|---|
| Model | EfficientNet-B0 |
| Image Size | 224 × 224 |
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Batch Size | 32 |
| Scheduler | Cosine Annealing |
| Loss | Cross Entropy with Label Smoothing |
| Early Stopping | Enabled |
| Pretrained Weights | ImageNet |

---

## 🔥 Explainable AI — Grad-CAM

The project uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide a visual explanation of the model's prediction.

Grad-CAM generates a heatmap showing the regions of the X-ray image that contributed most to the predicted class.

### Grad-CAM Workflow

```text
Chest X-Ray
     │
     ▼
EfficientNet-B0
     │
     ▼
Predicted Class
     │
     ▼
Gradient Calculation
     │
     ▼
Grad-CAM
     │
     ▼
Heatmap Generation
     │
     ▼
X-Ray + Heatmap
```

This provides an additional layer of interpretability to the classification result.

---

## 🌐 Streamlit Web Application

The project includes a Streamlit-based web application for interactive inference.

### Application Workflow

```text
Upload X-Ray
     │
     ▼
Image Preprocessing
     │
     ▼
EfficientNet-B0
     │
     ▼
Prediction
     │
     ├── Predicted Disease
     ├── Confidence Score
     └── Class Probabilities
     │
     ▼
Grad-CAM Visualization
```

### Run the Application

```bash
python -m streamlit run app.py
```

The application provides:

- 📤 X-ray image upload
- 🧠 Disease prediction
- 📊 Prediction confidence
- 📈 Class probability distribution
- 🔥 Grad-CAM visualization

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Sharif-Mahammad/Chest-XRay-Disease-Classification.git
```

### 2. Navigate to the Project Directory

```bash
cd Chest-XRay-Disease-Classification
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Step 1 — Prepare the Dataset

```bash
python prepare_data.py
```

### Step 2 — Train EfficientNet-B0

```bash
python train_compare.py --models efficientnet_b0
```

For a quick five-epoch training run:

```bash
python train_compare.py --models efficientnet_b0 --epochs 5 --batch-size 32 --num-workers 2
```

### Step 3 — Generate Grad-CAM

```bash
python gradcam.py --checkpoint outputs/efficientnet_b0_best.pt --image <image_path> --output outputs/gradcam_result.png
```

Example:

```bash
python gradcam.py --checkpoint outputs/efficientnet_b0_best.pt --image dataset/test/COVID19/sample.png --output outputs/gradcam_result.png
```

### Step 4 — Launch the Streamlit Application

```bash
python -m streamlit run app.py
```

---

## 📈 Model Performance

EfficientNet-B0 was trained and evaluated using the prepared three-class chest X-ray dataset.

The trained model achieved the following result on the provided test split:

| Metric | Score |
|---|---:|
| Test Accuracy | 100% |
| Test Images | 537 |

The evaluation was performed using the best validation checkpoint.

> ⚠️ The reported performance is specific to the provided dataset and test split. It should **not** be interpreted as clinical diagnostic accuracy or medical validation. Performance on external or real-world clinical data may differ.

---

## 📁 Output Files

After training, the project generates output files inside the `outputs/` directory.

```text
outputs/
│
├── efficientnet_b0_best.pt
├── efficientnet_b0_classification_report.json
├── model_comparison.csv
├── model_comparison.json
└── gradcam_test.png
```

### Model Checkpoint

```text
efficientnet_b0_best.pt
```

Contains the trained EfficientNet-B0 model weights and configuration.

### Classification Report

```text
efficientnet_b0_classification_report.json
```

Contains classification metrics generated during model evaluation.

### Grad-CAM Output

```text
gradcam_test.png
```

Contains a Grad-CAM visualization for a test X-ray image.

---

## ⚠️ Classification Behavior

This project performs **single-class classification**, not multi-label disease detection.

For each uploaded chest X-ray, the model selects the class with the highest predicted probability:

```text
COVID-19
Pneumonia
Tuberculosis
```

For example, if an X-ray contains visual characteristics associated with more than one condition, the model will still return **one predicted class** based on the highest model probability.

The current system does not support simultaneous predictions such as:

```text
COVID-19 + Pneumonia
```

True multi-label classification would require a different dataset labeling strategy, model output design, loss function, and training process.

---

## 📸 Sample Outputs

### 🔥 Grad-CAM

The project generates a Grad-CAM heatmap over the chest X-ray.

The generated output can be viewed from:

```text
outputs/gradcam_test.png
```

![Grad-CAM Output](outputs/gradcam_test.png)

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Model | EfficientNet-B0 |
| Computer Vision | Torchvision, Pillow |
| Explainable AI | Grad-CAM |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Web Application | Streamlit |
| Model Evaluation | Scikit-learn |
| Version Control | Git & GitHub |

---

## 📌 Complete Project Workflow

```text
                Dataset
                   │
                   ▼
            Data Preparation
                   │
                   ▼
          Image Preprocessing
                   │
                   ▼
           Data Augmentation
                   │
                   ▼
            EfficientNet-B0
                   │
                   ▼
             Model Training
                   │
                   ▼
              Validation
                   │
                   ▼
        Best Model Checkpoint
                   │
                   ▼
            Test Evaluation
                   │
                   ▼
             Grad-CAM
                   │
                   ▼
        Explainable Prediction
                   │
                   ▼
          Streamlit Web App
```

---

## 🔮 Future Improvements

- 🔹 Multi-label disease classification
- 🔹 Vision Transformer (ViT)
- 🔹 ConvNeXt architecture
- 🔹 Hyperparameter optimization
- 🔹 External dataset evaluation
- 🔹 Model quantization
- 🔹 ONNX deployment
- 🔹 FastAPI inference API
- 🔹 Cloud deployment
- 🔹 Improved model monitoring

---

## ⚠️ Disclaimer

This project is developed for **educational and research purposes only**.

It is not intended to replace professional medical diagnosis, clinical evaluation, or medical advice.

Predictions generated by this model should not be used to make healthcare decisions.

---

## 👨‍💻 Author

**Md. Sharif**

- GitHub: https://github.com/Sharif-Mahammad

---

## 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for more information.

---

⭐ If you found this project useful, consider giving it a star!
