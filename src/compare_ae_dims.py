import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score
import subprocess
import sys

# --- Configuration ---
DATA_DIR = '../data/'
MODEL_DIR = '../models/'
INPUT_FILE_X = 'X_test.csv'
INPUT_FILE_y = 'y_test.csv'
DIMS = [16, 32, 64]
RESULTS_FILE = os.path.join(MODEL_DIR, 'ae_dims_comparison.txt')

# --- 1. Train Models ---
print("Training models with different dimensions...")
for dim in DIMS:
    print(f"Training AE with hidden_dim={dim}...")
    subprocess.run([sys.executable, 'train_baseline_ae.py', '--hidden_dim', str(dim)], check=True)

# --- 2. Evaluation Setup ---
print("\nEvaluating models...")
X_test_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_X))
y_test_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_y))
X_test_tensor = torch.tensor(X_test_df.values, dtype=torch.float32)
# 1=Normal, 0=Anomaly -> 1=Anomaly, 0=Normal
y_test = 1 - y_test_df.iloc[:, 0].values

results = {}

# Define Autoencoder class dynamically or import it
# Since the class definition depends on HIDDEN_DIM_2 which is global in train_baseline_ae.py,
# it's safer to redefine it here or make it dynamic.
# Let's redefine it to be safe and flexible.

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

INPUT_DIM = 119
HIDDEN_DIM_1 = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for dim in DIMS:
    print(f"Evaluating AE with hidden_dim={dim}...")
    model_path = os.path.join(MODEL_DIR, f'baseline_ae_{dim}.pth')
    model = Autoencoder(INPUT_DIM, HIDDEN_DIM_1, dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    reconstruction_errors = []
    dataset = TensorDataset(X_test_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    with torch.no_grad():
        for data in loader:
            inputs = data[0].to(device)
            outputs = model(inputs)
            loss = torch.mean((outputs - inputs) ** 2, dim=1)
            reconstruction_errors.append(loss.cpu().numpy())
            
    errors = np.concatenate(reconstruction_errors)
    
    # Calculate AUC
    auc = roc_auc_score(y_test, errors)
    
    # Calculate Best F1
    best_f1 = 0
    thresholds = np.linspace(errors.min(), errors.max(), 100)
    for thresh in thresholds:
        y_pred = (errors > thresh).astype(int)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            
    results[dim] = {'AUC': auc, 'Best F1': best_f1}
    print(f"Dim {dim}: AUC={auc:.4f}, Best F1={best_f1:.4f}")

# --- 3. Save and Plot Results ---
print("\n--- Comparison Results ---")
with open(RESULTS_FILE, 'w') as f:
    f.write("Dimension,AUC,Best F1\n")
    for dim in DIMS:
        line = f"{dim},{results[dim]['AUC']:.4f},{results[dim]['Best F1']:.4f}"
        print(line)
        f.write(line + "\n")

# Plot
dims = list(results.keys())
aucs = [results[d]['AUC'] for d in dims]
f1s = [results[d]['Best F1'] for d in dims]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(dims, aucs, marker='o')
plt.title('AUC vs Latent Dimension')
plt.xlabel('Dimension')
plt.ylabel('AUC')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(dims, f1s, marker='o', color='orange')
plt.title('Best F1 vs Latent Dimension')
plt.xlabel('Dimension')
plt.ylabel('F1 Score')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'ae_dims_comparison.png'))
print(f"Comparison plot saved to {os.path.join(MODEL_DIR, 'ae_dims_comparison.png')}")
