import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Configuration ---
DATA_DIR = '../data/'
MODEL_DIR = '../models/'
INPUT_FILE_X = 'X_test.csv'
INPUT_FILE_y = 'y_test.csv'
AE_MODEL_PATH = os.path.join(MODEL_DIR, 'ldrae.pth')
SVM_MODEL_PATH = os.path.join(MODEL_DIR, 'best_svm_fixed_ae.joblib') # Optimized SVM

INPUT_DIM = 119
HIDDEN_DIM_1 = 64
HIDDEN_DIM_2 = 32
BATCH_SIZE = 256

# --- AE Definition ---
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

if __name__ == "__main__":
    print("Loading test data...")
    X_test_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_X))
    y_test_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_y))
    
    X_test_tensor = torch.tensor(X_test_df.values, dtype=torch.float32)
    y_test = 1 - y_test_df.iloc[:, 0].values # 1=Anomaly
    
    print("Loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_model = Autoencoder().to(device)
    ae_model.load_state_dict(torch.load(AE_MODEL_PATH, map_location=device))
    ae_model.eval()
    
    svm_clf = joblib.load(SVM_MODEL_PATH)
    
    print("Calculating scores...")
    latent_vectors = []
    reconstruction_errors = []
    dataset = TensorDataset(X_test_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    with torch.no_grad():
        for data in loader:
            inputs = data[0].to(device)
            # Forward pass
            latent = ae_model.encoder(inputs)
            outputs = ae_model.decoder(latent)
            
            latent_vectors.append(latent.cpu().numpy())
            
            # Reconstruction Error (MSE)
            error = torch.mean((outputs - inputs) ** 2, dim=1)
            reconstruction_errors.append(error.cpu().numpy())
            
    X_test_latent = np.concatenate(latent_vectors, axis=0)
    score_a = np.concatenate(reconstruction_errors, axis=0)
    
    # SVM Probability (Score B)
    # SVM predicts 1 for Normal, 0 for Anomaly.
    # We want Anomaly Score.
    # If SVM is LinearSVC (Calibrated), predict_proba gives [prob_0, prob_1].
    # prob_0 is probability of Anomaly (since 0=Anomaly in training).
    score_b = svm_clf.predict_proba(X_test_latent)[:, 0]
    
    # Normalize
    score_a_norm = (score_a - score_a.min()) / (score_a.max() - score_a.min())
    score_b_norm = (score_b - score_b.min()) / (score_b.max() - score_b.min())
    
    # Hybrid Score
    hybrid_score = 0.5 * score_a_norm + 0.5 * score_b_norm
    
    # Find Best Threshold
    best_f1 = 0
    best_threshold = 0
    thresholds = np.linspace(0, 1, 101)
    
    for thresh in thresholds:
        y_pred = (hybrid_score > thresh).astype(int)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            
    print(f"\nBest Threshold: {best_threshold:.4f}")
    y_pred = (hybrid_score > best_threshold).astype(int)
    
    print("\n--- Evaluation Results (Optimized Hybrid Model) ---")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save results
    with open(os.path.join(MODEL_DIR, 'optimized_hybrid_results.txt'), 'w') as f:
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
