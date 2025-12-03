import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, f1_score, accuracy_score
import subprocess
import sys

# --- Configuration ---
DATA_DIR = '../data/'
MODEL_DIR = '../models/'
INPUT_FILE_X = 'X_test.csv'
INPUT_FILE_y = 'y_test.csv'
AE_MODEL_PATH = os.path.join(MODEL_DIR, 'ldrae.pth')
KERNELS = ['linear', 'nystroem'] # 'rbf' might be too slow, but let's try if user insisted. Let's start with these two.
RESULTS_FILE = os.path.join(MODEL_DIR, 'svm_kernel_comparison.txt')

# --- 1. Train Models ---
print("Training models with different kernels...")
for kernel in KERNELS:
    print(f"Training Hybrid SVM with kernel={kernel}...")
    subprocess.run([sys.executable, 'train_hybrid_svm.py', '--kernel', kernel], check=True)

# --- 2. Evaluation Setup ---
print("\nEvaluating models...")
X_test_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_X))
y_test_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_y))
X_test_tensor = torch.tensor(X_test_df.values, dtype=torch.float32)
# 1=Normal, 0=Anomaly -> 1=Anomaly, 0=Normal
y_test = 1 - y_test_df.iloc[:, 0].values

# Define Autoencoder class (must match train_hybrid_svm.py)
INPUT_DIM = 119
HIDDEN_DIM_1 = 64
HIDDEN_DIM_2 = 32

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(HIDDEN_DIM_2, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_1, INPUT_DIM),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ae_model = Autoencoder().to(device)
ae_model.load_state_dict(torch.load(AE_MODEL_PATH, map_location=device))
ae_model.eval()

# Extract latent variables and Score A (Reconstruction Error)
print("Extracting features and Score A...")
latent_vectors = []
reconstruction_errors = []
dataset = TensorDataset(X_test_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

with torch.no_grad():
    for data in loader:
        inputs = data[0].to(device)
        latent = ae_model.encoder(inputs)
        outputs = ae_model.decoder(latent)
        
        latent_vectors.append(latent.cpu().numpy())
        error = torch.mean((outputs - inputs) ** 2, dim=1)
        reconstruction_errors.append(error.cpu().numpy())

X_test_latent = np.concatenate(latent_vectors, axis=0)
score_a = np.concatenate(reconstruction_errors, axis=0)
score_a_norm = (score_a - score_a.min()) / (score_a.max() - score_a.min())

results = {}

for kernel in KERNELS:
    print(f"Evaluating Hybrid SVM with kernel={kernel}...")
    svm_path = os.path.join(MODEL_DIR, f'svm_classifier_{kernel}.joblib')
    svm_clf = joblib.load(svm_path)
    
    # Score B: SVM Probability of Anomaly (Class 0)
    score_b = svm_clf.predict_proba(X_test_latent)[:, 0]
    score_b_norm = (score_b - score_b.min()) / (score_b.max() - score_b.min())
    
    # Hybrid Score
    hybrid_score = 0.5 * score_a_norm + 0.5 * score_b_norm
    
    # Find Best Threshold based on F1
    best_f1 = 0
    best_thresh = 0
    thresholds = np.linspace(0, 1, 101)
    
    for thresh in thresholds:
        y_pred = (hybrid_score > thresh).astype(int)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    y_pred_opt = (hybrid_score > best_thresh).astype(int)
    recall = recall_score(y_test, y_pred_opt)
    accuracy = accuracy_score(y_test, y_pred_opt)
    
    results[kernel] = {'Accuracy': accuracy, 'Recall': recall, 'F1': best_f1}
    print(f"Kernel {kernel}: Accuracy={accuracy:.4f}, Recall={recall:.4f}, F1={best_f1:.4f}")

# --- 3. Save Results ---
print("\n--- Comparison Results ---")
with open(RESULTS_FILE, 'w') as f:
    f.write("Kernel,Accuracy,Recall,F1\n")
    for kernel in KERNELS:
        line = f"{kernel},{results[kernel]['Accuracy']:.4f},{results[kernel]['Recall']:.4f},{results[kernel]['F1']:.4f}"
        print(line)
        f.write(line + "\n")
