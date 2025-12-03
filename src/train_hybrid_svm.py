import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
import numpy as np

import argparse
from sklearn.kernel_approximation import Nystroem

# --- Configuration ---
parser = argparse.ArgumentParser(description='Train Hybrid SVM')
parser.add_argument('--kernel', type=str, default='linear', choices=['linear', 'rbf', 'nystroem'], help='SVM kernel')
args = parser.parse_args()

DATA_DIR = '../data/'
MODEL_DIR = '../models/'
INPUT_FILE_X = 'X_train.csv'
INPUT_FILE_y = 'y_train.csv'
AE_MODEL_PATH = os.path.join(MODEL_DIR, 'ldrae.pth')
SVM_MODEL_PATH = os.path.join(MODEL_DIR, f'svm_classifier_{args.kernel}.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'svm_scaler.joblib')

# Hyperparameters (Must match LDR-AE training)
INPUT_DIM = 119
HIDDEN_DIM_1 = 64
HIDDEN_DIM_2 = 32

# --- 1. Define the Autoencoder Model (Same as LDR-AE) ---
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

# --- 2. Load Data and Model ---
if __name__ == "__main__":
    print("Loading data...")
    X_train_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_X))
    y_train_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_y))

    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train_df.values, dtype=torch.float32)

    print(f"Loading LDR-AE model from {AE_MODEL_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_model = Autoencoder().to(device)
    ae_model.load_state_dict(torch.load(AE_MODEL_PATH, map_location=device))
    ae_model.eval()

    # --- 3. Extract Latent Variables ---
    print("Extracting latent variables...")
    latent_vectors = []
    batch_size = 64
    dataset = TensorDataset(X_train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for data in loader:
            inputs = data[0].to(device)
            # Get latent representation (output of encoder)
            latent = ae_model.encoder(inputs)
            latent_vectors.append(latent.cpu().numpy())

    X_train_latent = np.concatenate(latent_vectors, axis=0)
    y_train = y_train_df.iloc[:, 0].values # Assuming label is in the first column

    print(f"Latent feature shape: {X_train_latent.shape}")

    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    # --- 4. Train SVM ---
    print(f"Training SVM classifier with {args.kernel} kernel...")

    if args.kernel == 'linear':
        # Use LinearSVC for speed, wrapped in CalibratedClassifierCV for probability estimation
        linear_svc = LinearSVC(dual=False, random_state=42)
        svm_clf = make_pipeline(StandardScaler(), CalibratedClassifierCV(linear_svc))
    elif args.kernel == 'nystroem':
        # Approximate RBF kernel using Nystroem
        nystroem = Nystroem(kernel='rbf', gamma=0.1, n_components=100, random_state=42)
        linear_svc = LinearSVC(dual=False, random_state=42)
        svm_clf = make_pipeline(StandardScaler(), nystroem, CalibratedClassifierCV(linear_svc))
    elif args.kernel == 'rbf':
        # True RBF Kernel (Slow!)
        # We use a subset or just hope it finishes? 
        # For 120k samples, it might take a while.
        # Let's use it but maybe with cache_size=1000
        svc = SVC(kernel='rbf', probability=True, cache_size=1000, random_state=42)
        svm_clf = make_pipeline(StandardScaler(), svc)

    svm_clf.fit(X_train_latent, y_train)

    # --- 5. Save SVM Model ---
    print("Saving SVM model...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(svm_clf, SVM_MODEL_PATH)
    # Note: The scaler is included in the pipeline, so we don't need to save it separately,
    # but saving the pipeline is sufficient.
    print(f"SVM model saved to {SVM_MODEL_PATH}")
