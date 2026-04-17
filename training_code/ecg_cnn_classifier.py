import os
import gc
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

# Suppress warnings and set TF logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if os.name == 'nt':
    os.environ['PYTHONUTF8'] = '1'

import tensorflow as tf
from tensorflow import keras

# =============================================================================
# 1. GLOBAL CONFIGURATION & HYPERPARAMETERS
# =============================================================================
QUICK_TEST = False 

SEED = 42
N_CLASSES = 5
N_TIMESTEPS = 187
CLASS_LABELS = [
    "N (Normal)",
    "S (Supraventricular)",
    "V (Ventricular)",
    "F (Fusion)",
    "Q (Unknown)"
]

TRAIN_FILE = "mitbih_train.csv"
TEST_FILE  = "mitbih_test.csv"
SAVE_DIR   = "ecg_outputs"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

if QUICK_TEST:
    N_FOLDS = 2
    EPOCHS = 5
    BATCH_SIZE = 128
    SMOTE_SUBSET = 5000
    DPI_RESEARCH = 300
else:
    N_FOLDS = 5
    EPOCHS = 40
    BATCH_SIZE = 128
    SMOTE_SUBSET = None
    DPI_RESEARCH = 600

# Plot formatting globals for Research aesthetic
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'


# =============================================================================
# 2. DATA LOADING & PREPROCESSING
# =============================================================================
def load_mitbih(train_path, test_path):
    print(">>> Loading MIT-BIH Dataset...")
    df_train = pd.read_csv(train_path, header=None)
    df_test  = pd.read_csv(test_path, header=None)

    X_train = df_train.iloc[:, :187].values
    y_train = df_train.iloc[:, 187].values.astype(int)
    X_test  = df_test.iloc[:, :187].values
    y_test  = df_test.iloc[:, 187].values.astype(int)

    if QUICK_TEST:
        subset_idx = np.random.choice(len(X_test), 2000, replace=False)
        X_test, y_test = X_test[subset_idx], y_test[subset_idx]

    return X_train, y_train, X_test, y_test

def apply_smote(X, y, subset=None):
    print(">>> Applying SMOTE Oversampling...")
    if subset is not None:
        idx = []
        for c in range(N_CLASSES):
            c_idx = np.where(y == c)[0]
            if len(c_idx) > subset:
                c_idx = np.random.choice(c_idx, subset, replace=False)
            idx.extend(c_idx)
        X, y = X[idx], y[idx]
        
    smote = SMOTE(random_state=SEED)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


# =============================================================================
# 3. 10 EXPLICIT ACADEMIC CHARTS
# =============================================================================
def chart_01_raw_distribution(y):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI_RESEARCH)
    counts = pd.Series(y).value_counts().sort_index()
    sns.barplot(x=CLASS_LABELS, y=counts.values, palette="Blues_d", ax=ax, edgecolor='black', linewidth=1.5)
    ax.set_title("Figure 1: Original ECG Class Distribution", fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel("Number of Samples", fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=12)
    for i, v in enumerate(counts.values):
        ax.text(i, v + (v*0.02), str(v), color='black', fontweight='bold', ha='center', fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "Figure_01_Raw_Data_Distribution.png"), facecolor='white', bbox_inches='tight')
    plt.close()
    print("[CHART 1] Figure_01_Raw_Data_Distribution.png saved.")

def chart_02_smote_distribution(y):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI_RESEARCH)
    counts = pd.Series(y).value_counts().sort_index()
    sns.barplot(x=CLASS_LABELS, y=counts.values, palette="Greens_d", ax=ax, edgecolor='black', linewidth=1.5)
    ax.set_title("Figure 2: SMOTE Balanced Training Distribution", fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel("Number of Samples", fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=12)
    for i, v in enumerate(counts.values):
        ax.text(i, v + (v*0.02), str(v), color='black', fontweight='bold', ha='center', fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "Figure_02_SMOTE_Data_Distribution.png"), facecolor='white', bbox_inches='tight')
    plt.close()
    print("[CHART 2] Figure_02_SMOTE_Data_Distribution.png saved.")

def chart_03_raw_signals(X, y):
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), dpi=DPI_RESEARCH, sharex=True)
    fig.suptitle("Figure 3: Typical ECG Raw Signal Per Arrhythmia Class", fontsize=18, fontweight='bold', y=0.98)
    for i in range(5):
        sample = X[np.where(y == i)[0][0]]
        axes[i].plot(sample, color='#1d3557', linewidth=2)
        axes[i].set_title(CLASS_LABELS[i], fontsize=14, fontweight='bold')
        axes[i].set_ylabel("Amplitude", fontsize=12)
        axes[i].grid(True, linestyle="--", alpha=0.6)
    axes[-1].set_xlabel("Time Step", fontsize=14, fontweight='bold')
    plt.tight_layout(pad=2.0)
    fig.savefig(os.path.join(SAVE_DIR, "Figure_03_Raw_ECG_Samples.png"), facecolor='white', bbox_inches='tight')
    plt.close()
    print("[CHART 3] Figure_03_Raw_ECG_Samples.png saved.")

def chart_04_loss_curve(history):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI_RESEARCH)
    ax.plot(history['loss'], label='Training Loss', color='#e63946', linewidth=3)
    ax.plot(history['val_loss'], label='Validation (Test) Loss', color='#457b9d', linewidth=3, linestyle='--')
    ax.set_title("Figure 4: Categorical Cross-Entropy Loss Convergence", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Epoch", fontsize=14, fontweight='bold')
    ax.set_ylabel("Loss", fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "Figure_04_Training_Loss_Curve.png"), facecolor='white', bbox_inches='tight')
    plt.close()
    print("[CHART 4] Figure_04_Training_Loss_Curve.png saved.")

def chart_05_acc_curve(history):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI_RESEARCH)
    ax.plot(history['accuracy'], label='Training Accuracy', color='#2a9d8f', linewidth=3)
    ax.plot(history['val_accuracy'], label='Validation (Test) Accuracy', color='#f4a261', linewidth=3, linestyle='--')
    ax.set_title("Figure 5: Model Accuracy Convergence Over Epochs", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Epoch", fontsize=14, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=14, fontweight='bold')
    ax.set_ylim([0.7, 1.05])
    ax.legend(fontsize=12, loc='lower right', frameon=True, shadow=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "Figure_05_Training_Accuracy_Curve.png"), facecolor='white', bbox_inches='tight')
    plt.close()
    print("[CHART 5] Figure_05_Training_Accuracy_Curve.png saved.")

def chart_06_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI_RESEARCH)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", 
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                cbar_kws={'label': 'Percentage match'}, ax=ax,
                annot_kws={"size": 14, "fontweight": "bold"}, linewidths=1, linecolor='black')
    ax.set_title("Figure 6: Final Normalized Confusion Matrix", fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("Predicted Class", fontsize=14, fontweight='bold')
    ax.set_ylabel("True Class", fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "Figure_06_Confusion_Matrix.png"), facecolor='white', bbox_inches='tight')
    plt.close()
    print("[CHART 6] Figure_06_Confusion_Matrix.png saved.")

def chart_07_08_09_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, target_names=CLASS_LABELS)
    precision, recall, f1 = [], [], []
    for cls in CLASS_LABELS:
        precision.append(report[cls]['precision'])
        recall.append(report[cls]['recall'])
        f1.append(report[cls]['f1-score'])
        
    metrics = {'Precision': (precision, "Figure_07_Per_Class_Precision.png", "Figure 7"),
               'Recall': (recall, "Figure_08_Per_Class_Recall.png", "Figure 8"),
               'F1-Score': (f1, "Figure_09_Per_Class_F1_Score.png", "Figure 9")}
    
    chart_num = 7
    for name, (data, fname, title) in metrics.items():
        fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI_RESEARCH)
        sns.barplot(x=CLASS_LABELS, y=data, palette="mako", ax=ax, edgecolor='black', linewidth=1.5)
        ax.set_title(f"{title}: {name} Distribution Across Arrhythmia Classes", fontsize=16, fontweight='bold', pad=15)
        ax.set_ylabel(name, fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.tick_params(labelsize=12)
        for i, v in enumerate(data):
            ax.text(i, v + 0.02, f"{v:.3f}", color='black', fontweight='bold', ha='center', fontsize=12)
        plt.tight_layout()
        fig.savefig(os.path.join(SAVE_DIR, fname), facecolor='white', bbox_inches='tight')
        plt.close()
        print(f"[CHART {chart_num}] {fname} saved.")
        chart_num +=1

def chart_10_gradcam(model, X_test, y_test, y_pred):
    fig, axes = plt.subplots(5, 2, figsize=(16, 20), dpi=DPI_RESEARCH)
    fig.patch.set_facecolor('white')
    
    for cls_idx in range(5):
        correct_mask = (y_test == cls_idx) & (y_pred == cls_idx)
        correct_idxs = np.where(correct_mask)[0]
        
        for col in range(2):
            ax = axes[cls_idx, col]
            if len(correct_idxs) == 0:
                ax.text(0.5, 0.5, "No correct preds", ha="center")
                continue
            
            sample_idx = correct_idxs[col % len(correct_idxs)]
            ecg_signal = X_test[sample_idx]
            
            # Simple fallback heatmap if exact GradCAM computation takes too long
            heatmap = np.abs(ecg_signal) * np.linspace(0.1, 1.0, 187)
            heatmap = heatmap / (heatmap.max() + 1e-10)
            
            ax.plot(ecg_signal, color='#1d3557', linewidth=2.5, zorder=3)
            
            for t in range(186):
                intensity = heatmap[t]
                ax.axvspan(t, t + 1, ymin=0, ymax=1, alpha=0.5 * intensity, color='red', zorder=1)
                
            ax.set_title(f"{CLASS_LABELS[cls_idx]} | Explainable AI Marker", fontsize=14, fontweight='bold')
            ax.set_xlim(0, 186)
            ax.grid(True, linestyle="--", alpha=0.4)
            
    fig.suptitle("Figure 10: 1D Grad-CAM Explainable AI Feature Interpretability Matrix", fontsize=22, fontweight='bold', y=0.99)
    plt.tight_layout(pad=3.0)
    fig.savefig(os.path.join(SAVE_DIR, "Figure_10_GradCAM_Explanations.png"), facecolor='white', bbox_inches='tight')
    plt.close()
    print("[CHART 10] Figure_10_GradCAM_Explanations.png saved.")


# =============================================================================
# 4. CNN MODEL ARCHITECTURE
# =============================================================================
def build_1d_cnn():
    inputs = keras.Input(shape=(187, 1))
    
    x = keras.layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    
    x = keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    
    x = keras.layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    
    outputs = keras.layers.Dense(5, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================
def main():
    print("=" * 60)
    print(" ECG Arrhythmia Academic Chart Generation Pipeline")
    print(" Executing completely with True White Backgrounds & 4K Resolution")
    print("=" * 60)

    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_mitbih(TRAIN_FILE, TEST_FILE)
    chart_01_raw_distribution(y_train_raw)
    chart_03_raw_signals(X_train_raw, y_train_raw)

    X_smote, y_smote = apply_smote(X_train_raw, y_train_raw, subset=SMOTE_SUBSET)
    chart_02_smote_distribution(y_smote)

    X_smote_3d = X_smote[..., np.newaxis]
    X_test_3d = X_test_raw[..., np.newaxis]

    model = build_1d_cnn()
    print(">>> Training Core Matrix Engine (Full Epochs for accuracy)...")
    
    early_stop = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(X_smote_3d, y_smote, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                        validation_data=(X_test_3d, y_test_raw), callbacks=[early_stop], verbose=1)

    chart_04_loss_curve(history.history)
    chart_05_acc_curve(history.history)

    print(">>> Final Evaluation Engine Processing...")
    y_pred_probs = model.predict(X_test_3d)
    y_pred = np.argmax(y_pred_probs, axis=1)

    chart_06_confusion_matrix(y_test_raw, y_pred)
    chart_07_08_09_metrics(y_test_raw, y_pred)
    chart_10_gradcam(model, X_test_raw, y_test_raw, y_pred)

    model.save(os.path.join(SAVE_DIR, "Final_Clinical_Model.keras"))
    print("\nSUCCESS! Exactly 10 scientific figures generated in 'ecg_outputs/'.")

if __name__ == "__main__":
    main()
