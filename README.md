# 🧠 Diabetes Prediction — Artificial Neural Network (Binary Classification)

> **Course:** Fundamentals of Computational Intelligence
> **Dataset:** Pima Indians Diabetes Dataset — 768 patients, 8 clinical features
---

## 📋 Project Description

This project implements a **feedforward Artificial Neural Network (ANN)** to perform binary classification on the Pima Indians Diabetes dataset. The goal is to predict whether a patient is diabetic (1) or non-diabetic (0) based on 8 clinical measurements.

### Objectives
- Apply neural network design principles to a real-world medical dataset.
- Handle class imbalance, missing-value imputation, and feature scaling in a clinical context.
- Evaluate model performance using accuracy, ROC-AUC, and a confusion matrix.
- Provide an interactive real-time inference terminal loop for live predictions.

### Key Results

| Metric        | Value  |
|---------------|--------|
| Test Accuracy | 75.86% |
| Test Loss     | 0.4408 |
| ROC-AUC       | 0.8712 |
| Epochs Run    | 21 (Early Stopping) |

---

## 🗂️ Project Structure

```
diabetes-ann/
│
├── main.py                        # Full training & inference pipeline
├── Dataset.csv                    # Pima Indians Diabetes dataset (local copy)
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation (this file)
│
└── outputs/
    └── diabetes_training_report.png   # Auto-generated training report (plots)
```

---

## ⚙️ ANN Architecture

```
Input Layer  →  8 clinical features
     ↓
Dense(32, activation='ReLU')
     ↓
Dropout(p = 0.30)
     ↓
Dense(16, activation='ReLU')
     ↓
Dense(1,  activation='Sigmoid')   →   Output: P(diabetic)
```

- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy
- **Regularization:** Dropout (p = 0.30)
- **Early Stopping:** Monitors `val_loss` with patience = 10

---

## 🔢 Dataset Features

| # | Feature                    | Description                                    |
|---|----------------------------|------------------------------------------------|
| 1 | `Pregnancies`              | Number of times pregnant                       |
| 2 | `Glucose`                  | Plasma glucose concentration (2-hr oral test)  |
| 3 | `BloodPressure`            | Diastolic blood pressure (mm Hg)               |
| 4 | `SkinThickness`            | Triceps skinfold thickness (mm)                |
| 5 | `Insulin`                  | 2-Hour serum insulin (μU/mL)                   |
| 6 | `BMI`                      | Body mass index (kg/m²)                        |
| 7 | `DiabetesPedigreeFunction` | Genetic risk score based on family history     |
| 8 | `Age`                      | Age in years                                   |
|   | `Outcome` *(target)*       | 1 = Diabetic, 0 = Non-diabetic                 |

> **Note:** Zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are biologically impossible and are imputed with the column **median** before training.

---

## 🛠️ Prerequisites & Dependencies

- **Python** 3.9 or higher
- **pip** (Python package manager)

### Required Libraries

```
tensorflow  >= 2.15.0
scikit-learn >= 1.4.0
pandas      >= 2.1.0
numpy       >= 1.26.0
matplotlib  >= 3.8.0
```

---

## 🚀 Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/diabetes-ann.git
cd diabetes-ann
```

### 2. (Recommended) Create a Virtual Environment

```bash
# Create environment
python -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline

```bash
python main.py
```

The script will:
1. Load `Dataset.csv` from the current directory (falls back to URL if not found).
2. Preprocess features and split data into Train / Validation / Test sets.
3. Build, train, and evaluate the ANN.
4. Save `diabetes_training_report.png` with all training plots.
5. Launch an **interactive terminal loop** for real-time patient predictions.

### 5. Real-Time Inference

After training completes, the terminal will prompt you to enter patient values one by one. Type `exit` or press `Ctrl+C` to quit.

---

## 📊 Training Report

The pipeline automatically generates a 6-panel visual report (`diabetes_training_report.png`):

| Panel | Description |
|-------|-------------|
| Training vs Validation Accuracy | Learning curve over epochs |
| Training vs Validation Loss | Convergence behavior |
| ROC Curve | AUC = 0.871 |
| Confusion Matrix | Per-class predictions on the test set |
| Feature Importance | 1st-layer weight magnitudes |
| Model Configuration | Architecture & hyperparameter summary |

---

## 📄 License

This project is submitted as academic coursework and is intended for **educational purposes only**.

> ⚕️ **Medical Disclaimer:** Predictions made by this model are for educational purposes only. They should never be used as a substitute for professional medical advice, diagnosis, or treatment.

---

*Generated with Python · TensorFlow · scikit-learn · Matplotlib*
