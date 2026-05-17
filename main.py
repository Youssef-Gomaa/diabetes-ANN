# =============================================================================
# Diabetes Prediction — Artificial Neural Network (Binary Classification)
# Course: Fundamentals of Computational Intelligence
# Dataset: Pima Indians Diabetes Dataset (768 patients, 8 clinical features)
#ُ Eng: Hassan El-Sayed
# =============================================================================
# PIPELINE OVERVIEW
# ─────────────────
#  1. Load dataset  →  from local CSV (auto-fallback to URL if not found)
#  2. Explore data  →  shape, class balance, missing-value check
#  3. Preprocess    →  zero-value imputation, StandardScaler, train/val/test split
#  4. Build ANN     →  Input → Dense(32,ReLU) → Dense(16,ReLU) → Dense(1,Sigmoid)
#  5. Train         →  Adam optimizer, Binary Cross-Entropy, EarlyStopping
#  6. Evaluate      →  accuracy, classification report, confusion matrix
#  7. Plot          →  training curves + confusion matrix → saved as PNG
#  8. Inference     →  interactive terminal loop for real-time predictions
# =============================================================================
 
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import re
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TensorFlow info messages
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                        # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
 
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: REPRODUCIBILITY SEED
# ─────────────────────────────────────────────────────────────────────────────
# A fixed seed guarantees identical results on every run — crucial for
# academic work where you need to reproduce your own experiments.
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
 
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: LOAD THE PIMA INDIANS DIABETES DATASET
# ─────────────────────────────────────────────────────────────────────────────
# The dataset describes 768 female patients of Pima Indian heritage.
# Each row is one patient; the last column is the outcome (1=diabetic, 0=not).
#
# Feature columns (8 total):
#   1. Pregnancies          – Number of times pregnant
#   2. Glucose              – Plasma glucose concentration (2-hour oral test)
#   3. BloodPressure        – Diastolic blood pressure (mm Hg)
#   4. SkinThickness        – Triceps skinfold thickness (mm)
#   5. Insulin              – 2-Hour serum insulin (mu U/ml)
#   6. BMI                  – Body mass index (weight/height²)
#   7. DiabetesPedigreeFunction – Genetic risk score based on family history
#   8. Age                  – Age in years
#   Target: Outcome         – 1 = Diabetic, 0 = Not diabetic
 
COLUMN_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
 
LOCAL_CSV    = "Dataset.csv"
DATASET_URL  = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
 
print("=" * 65)
print("  Diabetes Prediction — ANN Binary Classifier")
print("  Fundamentals of Computational Intelligence")
print("=" * 65)
 
print("\n[1/6] Loading dataset...")
 
# FIX: Prefer the local CSV (which already has a header row).
# Only fall back to the URL if the local file is missing.
# When reading from the URL, the CSV also already has a header,
# so we never blindly overwrite column names.
if os.path.exists(LOCAL_CSV):
    df = pd.read_csv(LOCAL_CSV)
    # Normalise column names in case of minor whitespace differences
    df.columns = [c.strip() for c in df.columns]
    print(f"      Source  : Local file ({LOCAL_CSV})")
else:
    try:
        df = pd.read_csv(DATASET_URL)
        df.columns = [c.strip() for c in df.columns]
        print(f"      Source  : URL  ({DATASET_URL[:55]}...)")
    except Exception as e:
        raise RuntimeError(
            f"Could not load dataset from URL ({e}). "
            f"Place '{LOCAL_CSV}' in the same directory as this script."
        )
 
# Validate that the expected columns are present
missing_cols = [c for c in COLUMN_NAMES if c not in df.columns]
if missing_cols:
    raise ValueError(f"Dataset is missing expected columns: {missing_cols}")
 
print(f"      Shape   : {df.shape[0]} rows × {df.shape[1]} columns")
 
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: EXPLORATORY DATA OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Exploring the dataset...")
 
# Class distribution
class_counts = df['Outcome'].value_counts()
class_pct    = df['Outcome'].value_counts(normalize=True) * 100
print(f"      Patients (total): {len(df)}")
print(f"      Non-diabetic (0) : {class_counts[0]}  ({class_pct[0]:.1f}%)")
print(f"      Diabetic     (1) : {class_counts[1]}  ({class_pct[1]:.1f}%)")
 
# The Pima dataset uses 0 as a placeholder for "missing" in several columns.
# Biologically, a Glucose, BMI, or BloodPressure of 0 is impossible.
ZERO_IMPOSSIBLE = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
zero_counts = (df[ZERO_IMPOSSIBLE] == 0).sum()
print(f"\n      Zero-value counts (biologically impossible → treated as NaN):")
for col, cnt in zero_counts.items():
    if cnt > 0:
        print(f"        {col:<28}: {cnt} rows")
 
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Preprocessing...")
 
# Step 5a: Replace impossible zeros with the column MEDIAN
# We use the MEDIAN (not mean) because it is robust to outliers.
# We compute the median on the full dataset here for simplicity
# (acceptable for a course project).
df_clean = df.copy()
# Store training medians for use during inference imputation later
TRAIN_MEDIANS = {}
for col in ZERO_IMPOSSIBLE:
    median_val = df_clean[col].replace(0, np.nan).median()
    TRAIN_MEDIANS[col] = median_val
    df_clean[col] = df_clean[col].replace(0, median_val)
 
print(f"      Missing-zero imputation: done (replaced with column median)")
 
# Step 5b: Separate features (X) and target label (y)
FEATURE_COLS = COLUMN_NAMES[:-1]     # First 8 columns = features
X = df_clean[FEATURE_COLS].values    # Shape: (768, 8) — NumPy array
y = df_clean['Outcome'].values       # Shape: (768,)  — binary labels
 
print(f"      Features (X) shape : {X.shape}")
print(f"      Target   (y) shape : {y.shape}")
 
# Step 5c: Train / Validation / Test split
# We create THREE splits:
#   Train (70%) → used to update the model's weights
#   Val   (15%) → monitored during training to catch overfitting (not used for weight updates)
#   Test  (15%) → only evaluated ONCE at the end — the model's "final exam"
#
# stratify=y → ensures both splits have the same class ratio as the full dataset
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)
 
print(f"\n      Train set      : {X_train.shape[0]} samples  (70%)")
print(f"      Validation set : {X_val.shape[0]}  samples  (15%)")
print(f"      Test set       : {X_test.shape[0]}  samples  (15%)")
 
# Step 5d: Feature scaling with StandardScaler
# Neural networks train faster and more stably when all features share a similar
# numeric range. StandardScaler transforms each feature to:
#   Mean = 0, Standard Deviation = 1
#
# CRITICAL RULE: fit the scaler ONLY on training data.
# Fitting on test/val data would be "data leakage" — the model would
# indirectly see information about those sets before evaluation.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # Fit & transform training data
X_val   = scaler.transform(X_val)         # Transform only (no fit)
X_test  = scaler.transform(X_test)        # Transform only (no fit)
 
print(f"      StandardScaler : applied  (mean≈0, std≈1 per feature)")
 
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: BUILD THE ANN MODEL
# ─────────────────────────────────────────────────────────────────────────────
# Architecture:
#
#   INPUT (8 features: Glucose, BMI, Age, ...)
#               ↓
#   ┌───────────────────────┐
#   │  Dense(32, ReLU)      │  ← Hidden Layer 1
#   └───────────────────────┘
#               ↓
#   ┌───────────────────────┐
#   │  Dropout(0.3)         │  ← Regularization: randomly disables 30% of neurons
#   └───────────────────────┘   during training to prevent overfitting
#               ↓
#   ┌───────────────────────┐
#   │  Dense(16, ReLU)      │  ← Hidden Layer 2
#   └───────────────────────┘
#               ↓
#   ┌───────────────────────┐
#   │  Dense(1, Sigmoid)    │  ← Output Layer (probability of diabetes)
#   └───────────────────────┘
#               ↓
#    P(Diabetic) — if > 0.5 → Class 1 (Diabetic)
#                  if ≤ 0.5 → Class 0 (Not Diabetic)
print("\n[4/6] Building ANN model...")
 
n_features = X_train.shape[1]   # = 8
 
model = Sequential(name="Diabetes_ANN")
 
model.add(Input(shape=(n_features,)))
 
# ── Hidden Layer 1 ──────────────────────────────────────────────────────────
model.add(Dense(32, activation='relu', name='hidden_layer_1'))
 
# ── Dropout Regularization ───────────────────────────────────────────────────
# 30% of neurons are randomly disabled at each training step.
# At inference time, Dropout is automatically turned OFF — all neurons active.
model.add(Dropout(0.30, name='dropout_1'))
 
# ── Hidden Layer 2 ──────────────────────────────────────────────────────────
model.add(Dense(16, activation='relu', name='hidden_layer_2'))
 
# ── Output Layer ─────────────────────────────────────────────────────────────
model.add(Dense(1, activation='sigmoid', name='output_layer'))
 
# ── Compile ───────────────────────────────────────────────────────────────────
# optimizer='adam'            → Adaptive Moment Estimation. Adjusts the
#                               learning rate automatically for each parameter.
#                               Best general-purpose optimizer.
# loss='binary_crossentropy'  → Standard loss for binary (0/1) classification.
#                               L = -(y·log(ŷ) + (1-y)·log(1-ŷ))
#                               Penalises confident wrong predictions heavily.
# metrics=['accuracy']        → Track % of correct predictions during training.
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
 
model.summary()
 
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: TRAINING
# ─────────────────────────────────────────────────────────────────────────────
# EarlyStopping:
#   Watches val_loss. If it fails to improve for 'patience' consecutive epochs,
#   training stops early. This saves time and prevents overfitting.
#   restore_best_weights=True → rolls back weights to the best epoch seen.
print("\n[5/6] Training the model...")
 
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)
 
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,             # Maximum training epochs
    batch_size=16,          # Small batch size suits small datasets
    callbacks=[early_stop],
    verbose=1
)
 
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: EVALUATION ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────
# The test set was never used during training or validation.
# This gives a realistic estimate of how the model would perform on new patients.
print("\n[6/6] Evaluating on unseen test set...")
 
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
 
# Raw sigmoid probabilities → binary labels
y_pred_proba = model.predict(X_test, verbose=0).flatten()
y_pred       = (y_pred_proba >= 0.5).astype(int)
 
# AUC-ROC: 0.5 = random guessing, 1.0 = perfect classifier
auc_score = roc_auc_score(y_test, y_pred_proba)
 
print(f"\n  ┌───────────────────────────────────────┐")
print(f"  │  Test Loss     : {test_loss:.4f}                │")
print(f"  │  Test Accuracy : {test_acc * 100:.2f}%               │")
print(f"  │  ROC-AUC Score : {auc_score:.4f}                │")
print(f"  └───────────────────────────────────────┘")
 
print("\n  Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Non-Diabetic (0)", "Diabetic (1)"]
))
 
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating training report plot...")
 
# ── Colour palette (matplotlib hex colours — used ONLY for plotting) ─────────
PLT_BG      = "#0D1117"
PLT_PANEL   = "#161B22"
PLT_GRID    = "#21262D"
PLT_TEXT    = "#C9D1D9"
PLT_BLUE    = "#58A6FF"
PLT_ORANGE  = "#F78166"
PLT_GREEN   = "#3FB950"
PLT_PURPLE  = "#BC8CFF"
PLT_YELLOW  = "#E3B341"
PLT_BORDER  = "#30363D"
 
plt.rcParams.update({
    'font.family': 'monospace',
    'text.color':       PLT_TEXT,
    'axes.labelcolor':  PLT_TEXT,
    'xtick.color':      PLT_TEXT,
    'ytick.color':      PLT_TEXT,
    'axes.edgecolor':   PLT_BORDER,
})
 
epochs_ran = range(1, len(history.history['loss']) + 1)
 
fig = plt.figure(figsize=(16, 10), facecolor=PLT_BG)
fig.suptitle(
    "Diabetes Prediction — ANN Training Report",
    fontsize=17, fontweight='bold', color=PLT_TEXT, y=0.97
)
 
gs = gridspec.GridSpec(
    2, 3, figure=fig,
    hspace=0.50, wspace=0.38,
    left=0.07, right=0.97,
    top=0.90, bottom=0.08
)
 
def style_ax(ax, title):
    ax.set_facecolor(PLT_PANEL)
    ax.set_title(title, color=PLT_TEXT, fontsize=10.5, fontweight='bold', pad=9)
    ax.grid(True, color=PLT_GRID, linewidth=0.7, linestyle='--')
    for spine in ax.spines.values():
        spine.set_color(PLT_BORDER); spine.set_linewidth(0.6)
 
# ── Plot 1: Accuracy Curves ──────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Training vs Validation Accuracy")
ax1.plot(epochs_ran, history.history['accuracy'],
         color=PLT_BLUE, lw=2, label='Train Accuracy')
ax1.plot(epochs_ran, history.history['val_accuracy'],
         color=PLT_ORANGE, lw=2, linestyle='--', label='Val Accuracy')
ax1.axhline(test_acc, color=PLT_GREEN, lw=1.2, linestyle=':',
            label=f'Test Acc: {test_acc:.3f}')
ax1.set_xlabel("Epoch", fontsize=9)
ax1.set_ylabel("Accuracy", fontsize=9)
ax1.set_ylim([0.4, 1.05])
ax1.legend(facecolor=PLT_BG, edgecolor=PLT_BORDER, labelcolor=PLT_TEXT, fontsize=8)
 
# ── Plot 2: Loss Curves ──────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Training vs Validation Loss")
ax2.plot(epochs_ran, history.history['loss'],
         color=PLT_BLUE, lw=2, label='Train Loss')
ax2.plot(epochs_ran, history.history['val_loss'],
         color=PLT_ORANGE, lw=2, linestyle='--', label='Val Loss')
ax2.axhline(test_loss, color=PLT_GREEN, lw=1.2, linestyle=':',
            label=f'Test Loss: {test_loss:.3f}')
ax2.set_xlabel("Epoch", fontsize=9)
ax2.set_ylabel("Binary Cross-Entropy Loss", fontsize=9)
ax2.legend(facecolor=PLT_BG, edgecolor=PLT_BORDER, labelcolor=PLT_TEXT, fontsize=8)
 
# ── Plot 3: ROC Curve ────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, f"ROC Curve  (AUC = {auc_score:.3f})")
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax3.plot(fpr, tpr, color=PLT_PURPLE, lw=2.5, label=f'AUC = {auc_score:.3f}')
ax3.plot([0, 1], [0, 1], color=PLT_GRID, lw=1.2, linestyle='--',
         label='Random (AUC=0.5)')
ax3.fill_between(fpr, tpr, alpha=0.10, color=PLT_PURPLE)
ax3.set_xlabel("False Positive Rate", fontsize=9)
ax3.set_ylabel("True Positive Rate", fontsize=9)
ax3.legend(facecolor=PLT_BG, edgecolor=PLT_BORDER, labelcolor=PLT_TEXT, fontsize=8)
 
# ── Plot 4: Confusion Matrix ─────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor(PLT_PANEL)
ax4.set_title("Confusion Matrix (Test Set)", color=PLT_TEXT,
              fontsize=10.5, fontweight='bold', pad=9)
for spine in ax4.spines.values():
    spine.set_color(PLT_BORDER); spine.set_linewidth(0.6)
 
cm = confusion_matrix(y_test, y_pred)
im = ax4.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
 
labels = ['Non-Diabetic', 'Diabetic']
ax4.set_xticks([0, 1]); ax4.set_xticklabels(labels, color=PLT_TEXT, fontsize=8)
ax4.set_yticks([0, 1]); ax4.set_yticklabels(labels, color=PLT_TEXT, fontsize=8)
ax4.set_xlabel("Predicted", fontsize=9, color=PLT_TEXT)
ax4.set_ylabel("Actual",    fontsize=9, color=PLT_TEXT)
thresh = cm.max() / 2.0
for i in range(2):
    for j in range(2):
        ax4.text(j, i, f"{cm[i,j]}",
                 ha='center', va='center', fontsize=16, fontweight='bold',
                 color='white' if cm[i,j] > thresh else 'black')
 
# ── Plot 5: Feature Importance (via weight magnitudes) ───────────────────────
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5, "Input Feature Importance\n(1st-layer weight magnitude)")
 
# The magnitude of incoming weights to the first hidden layer gives a rough
# measure of how much influence each feature has on the network.
w1 = model.get_layer('hidden_layer_1').get_weights()[0]   # shape (8, 32)
feature_importance = np.abs(w1).mean(axis=1)              # mean across neurons
feat_order = np.argsort(feature_importance)
 
colors_bar = [PLT_YELLOW if v >= np.percentile(feature_importance, 60) else PLT_BLUE
              for v in feature_importance[feat_order]]
ax5.barh(
    [FEATURE_COLS[i] for i in feat_order],
    feature_importance[feat_order],
    color=colors_bar, edgecolor=PLT_BORDER, linewidth=0.5
)
ax5.set_xlabel("Mean |Weight|", fontsize=9)
ax5.tick_params(labelsize=8)
 
# ── Plot 6: Model Summary Card ───────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(PLT_PANEL)
ax6.set_title("Model Configuration", color=PLT_TEXT,
              fontsize=10.5, fontweight='bold', pad=9)
ax6.axis('off')
for spine in ax6.spines.values():
    spine.set_color(PLT_BORDER)
 
summary = [
    ("Dataset",         "Pima Indians Diabetes (768 patients)"),
    ("Features",        "8 clinical measurements"),
    ("Split",           "Train 70% / Val 15% / Test 15%"),
    ("Architecture",    "Input → 32 → Dropout → 16 → 1"),
    ("Activations",     "ReLU · ReLU · Sigmoid"),
    ("Optimizer",       "Adam"),
    ("Loss",            "Binary Cross-Entropy"),
    ("Regularization",  "Dropout (p=0.30)"),
    ("Epochs run",      f"{len(epochs_ran)}  (max 150 + EarlyStopping)"),
    ("Test Accuracy",   f"{test_acc * 100:.2f}%"),
    ("Test Loss",       f"{test_loss:.4f}"),
    ("ROC-AUC",         f"{auc_score:.4f}"),
]
 
y_pos = 0.95
for label, value in summary:
    ax6.text(0.02, y_pos, f"{label}:", color=PLT_PURPLE,
             fontsize=8, fontweight='bold', transform=ax6.transAxes, va='top')
    ax6.text(0.42, y_pos, value, color=PLT_TEXT,
             fontsize=8, transform=ax6.transAxes, va='top')
    y_pos -= 0.077
 
# ── Save ─────────────────────────────────────────────────────────────────────
out_path = "diabetes_training_report.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=PLT_BG)
plt.close()
print(f"  Plot saved → {out_path}")
 
print("\n" + "=" * 65)
print(f"  Done! Test Accuracy = {test_acc*100:.2f}% | AUC = {auc_score:.4f}")
print(f"  Open '{out_path}' in VS Code to view the training report.")
print("=" * 65)
 
# =============================================================================
# SECTION 10: REAL-TIME INFERENCE
# =============================================================================
 
# ─────────────────────────────────────────────────────────────────────────────
# ANSI COLOUR / STYLE CONSTANTS  (terminal display only — NOT matplotlib)
# Works on any modern terminal (Linux, macOS, Windows 10+ with VT mode).
# ─────────────────────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
 
ANSI_RED     = "\033[91m"
ANSI_GREEN   = "\033[92m"
ANSI_YELLOW  = "\033[93m"
ANSI_CYAN    = "\033[96m"
ANSI_WHITE   = "\033[97m"
ANSI_MAGENTA = "\033[95m"
 
BG_RED   = "\033[41m"
BG_GREEN = "\033[42m"
 
 
# ─────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def _box(lines: list, width: int = 62, colour: str = "") -> str:
    """Return a formatted ASCII box containing *lines* of text."""
    if not colour:
        colour = ANSI_CYAN
    top    = f"{colour}╔{'═' * width}╗{RESET}"
    bottom = f"{colour}╚{'═' * width}╝{RESET}"
    body   = []
    for line in lines:
        # Strip existing ANSI codes for length calculation
        plain = re.sub(r'\033\[[0-9;]*m', '', line)
        padding = width - len(plain) - 2       # 2 for the side spaces
        body.append(f"{colour}║{RESET} {line}{' ' * max(padding, 0)} {colour}║{RESET}")
    return "\n".join([top] + body + [bottom])
 
 
def _separator(char: str = "─", width: int = 64, colour: str = "") -> str:
    if not colour:
        colour = DIM + ANSI_WHITE
    return f"{colour}{char * width}{RESET}"
 
 
def _prompt_float(label: str, lo: float, hi: float) -> float:
    """
    Prompt the user for a single float value.
    Keeps asking until the input is a valid number within [lo, hi].
    """
    while True:
        raw = input(f"  {ANSI_CYAN}{label:<32}{RESET}: ").strip()
        if raw == "":
            print(f"  {ANSI_YELLOW}⚠  Input cannot be empty.{RESET}")
            continue
        try:
            val = float(raw)
        except ValueError:
            print(f"  {ANSI_YELLOW}⚠  '{raw}' is not a valid number. Try again.{RESET}")
            continue
        if not (lo <= val <= hi):
            print(f"  {ANSI_YELLOW}⚠  Value must be between {lo} and {hi}.{RESET}")
            continue
        return val
 
 
# ─────────────────────────────────────────────────────────────────────────────
# FIELD DEFINITIONS
# Each tuple: (display_label, internal_name, min_valid, max_valid, hint)
# ─────────────────────────────────────────────────────────────────────────────
INFERENCE_FIELDS = [
    ("Pregnancies",           "Pregnancies",              0,   20,  "0 – 20"),
    ("Glucose (mg/dL)",       "Glucose",                  0,  300,  "0 – 300"),
    ("Blood Pressure (mmHg)", "BloodPressure",            0,  150,  "0 – 150"),
    ("Skin Thickness (mm)",   "SkinThickness",            0,  100,  "0 – 100"),
    ("Insulin (μU/mL)",       "Insulin",                  0,  900,  "0 – 900"),
    ("BMI",                   "BMI",                      0,   80,  "0 – 80"),
    ("Diabetes Pedigree Fn",  "DiabetesPedigreeFunction", 0,    3,  "0.0 – 3.0"),
    ("Age (years)",           "Age",                     18,  120,  "18 – 120"),
]
 
 
def _risk_bar(prob: float, width: int = 40, colour: str = "") -> str:
    """Return a coloured ASCII progress bar representing diabetes probability."""
    if not colour:
        colour = ANSI_GREEN
    filled = int(prob * width)
    empty  = width - filled
    bar    = f"{colour}{'█' * filled}{DIM}{'░' * empty}{RESET}"
    return f"{DIM}Risk   [{RESET}{bar}{DIM}]{RESET}  {colour}{prob * 100:.1f}%{RESET}"
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN INFERENCE LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_inference_loop(model, scaler, feature_cols: list) -> None:
    """
    Interactive terminal loop that accepts patient data, preprocesses it
    with the already-fitted scaler, runs the trained model, and prints
    a styled prediction result.
 
    Parameters
    ----------
    model        : trained Keras model
    scaler       : fitted StandardScaler (from training pipeline)
    feature_cols : ordered list of feature column names (length 8)
    """
 
    print("\n")
    print(_separator("═", 64, colour=BOLD + ANSI_CYAN))
    print(_box(
        [
            f"  {BOLD}{ANSI_WHITE}  🩺  DIABETES RISK PREDICTOR — Real-Time Inference{RESET}",
            f"{DIM}  Enter patient values to get an instant prediction.{RESET}",
            f"{DIM}  Type  {BOLD}exit{RESET}{DIM}  or press  Ctrl-C  to quit.{RESET}",
        ],
        width=62, colour=ANSI_CYAN
    ))
    print(_separator("═", 64, colour=BOLD + ANSI_CYAN))
 
    case_number = 1
 
    while True:
        # ── Section header ────────────────────────────────────────────────
        print(f"\n  {BOLD}{ANSI_MAGENTA}▌ Case #{case_number}  —  Patient Data Input{RESET}")
        print(_separator("─", 64))
 
        # Print hint table
        print(f"  {DIM}{'Field':<32}  {'Valid range':<14}{RESET}")
        print(f"  {DIM}{'─'*32}  {'─'*14}{RESET}")
        for label, _, lo, hi, hint in INFERENCE_FIELDS:
            print(f"  {DIM}{label:<32}  {hint:<14}{RESET}")
        print(_separator("─", 64))
 
        # ── Collect one value per feature ─────────────────────────────────
        values = {}
        try:
            for label, name, lo, hi, _ in INFERENCE_FIELDS:
                values[name] = _prompt_float(label, lo, hi)
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  {ANSI_YELLOW}Session ended by user.{RESET}\n")
            break
 
        # ── Assemble feature vector in training order ─────────────────────
        raw_input = np.array([[values[col] for col in feature_cols]],
                             dtype=np.float64)   # shape (1, 8)
 
        # ── Apply the SAME imputation logic used during training ──────────
        # FIX: Use the actual training-set medians (stored in TRAIN_MEDIANS)
        # instead of scaler.mean_ (which is the post-scaling mean ≈ 0, not
        # a meaningful substitute for missing raw feature values).
        ZERO_IMPOSSIBLE_INF = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for i, col in enumerate(feature_cols):
            if col in ZERO_IMPOSSIBLE_INF and raw_input[0, i] == 0.0:
                raw_input[0, i] = TRAIN_MEDIANS[col]
 
        # ── Scale with the TRAINING scaler (no re-fitting) ────────────────
        scaled_input = scaler.transform(raw_input)          # shape (1, 8)
 
        # ── Forward pass ──────────────────────────────────────────────────
        prob = float(model.predict(scaled_input, verbose=0).flatten()[0])
        predicted_class = 1 if prob >= 0.5 else 0
        confidence_pct  = prob * 100 if predicted_class == 1 else (1 - prob) * 100
 
        # ── Build result display ──────────────────────────────────────────
        print(f"\n  {BOLD}{ANSI_WHITE}{'─'*60}{RESET}")
 
        if predicted_class == 1:                            # ── DIABETIC ──
            risk_bar   = _risk_bar(prob, width=40, colour=ANSI_RED)
            result_box = _box(
                [
                    f"  {BOLD}{ANSI_RED}🚨  PREDICTION :  HIGH DIABETES RISK{RESET}",
                    f"",
                    f"  {ANSI_WHITE}Confidence   : {BOLD}{ANSI_RED}{confidence_pct:>6.2f}%{RESET}",
                    f"  {ANSI_WHITE}Raw Prob.    : {BOLD}{ANSI_RED}{prob:.4f}{RESET}  {DIM}(threshold 0.50){RESET}",
                    f"  {ANSI_WHITE}Risk Level   : {BOLD}{ANSI_RED}POSITIVE — Consult a physician{RESET}",
                    f"",
                    f"  {risk_bar}",
                ],
                width=62, colour=ANSI_RED
            )
        else:                                               # ── HEALTHY ──
            risk_bar   = _risk_bar(prob, width=40, colour=ANSI_GREEN)
            result_box = _box(
                [
                    f"  {BOLD}{ANSI_GREEN}✅  PREDICTION :  LOW DIABETES RISK{RESET}",
                    f"",
                    f"  {ANSI_WHITE}Confidence   : {BOLD}{ANSI_GREEN}{confidence_pct:>6.2f}%{RESET}",
                    f"  {ANSI_WHITE}Raw Prob.    : {BOLD}{ANSI_GREEN}{prob:.4f}{RESET}  {DIM}(threshold 0.50){RESET}",
                    f"  {ANSI_WHITE}Risk Level   : {BOLD}{ANSI_GREEN}NEGATIVE — No immediate concern{RESET}",
                    f"",
                    f"  {risk_bar}",
                ],
                width=62, colour=ANSI_GREEN
            )
 
        print(result_box)
 
        # ── Feature echo table ────────────────────────────────────────────
        print(f"\n  {BOLD}{ANSI_WHITE}  Patient Profile Summary{RESET}")
        print(_separator("─", 64))
        for col, val in zip(feature_cols, raw_input[0]):
            bar_len = min(int(val / 5), 30)     # rough visual bar
            print(f"  {ANSI_CYAN}{col:<28}{RESET}  {ANSI_WHITE}{val:>8.2f}{RESET}  "
                  f"{DIM}{'▪' * bar_len}{RESET}")
        print(_separator("─", 64))
 
        # ── Medical disclaimer ────────────────────────────────────────────
        print(f"  {DIM}⚕  This prediction is for educational purposes only.")
        print(f"      Always consult a qualified healthcare professional.{RESET}\n")
 
        # ── Continue prompt ───────────────────────────────────────────────
        try:
            again = input(
                f"  {ANSI_YELLOW}Run another prediction? [Y/n]: {RESET}"
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  {ANSI_YELLOW}Session ended by user.{RESET}\n")
            break
 
        if again in ("n", "no", "exit", "quit", "q"):
            print(f"\n  {ANSI_GREEN}Thank you for using the Diabetes Risk Predictor.{RESET}\n")
            break
 
        case_number += 1
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT — called after training completes
# ─────────────────────────────────────────────────────────────────────────────
run_inference_loop(model, scaler, FEATURE_COLS)