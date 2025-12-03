import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import joblib
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Configuration ---
DATA_DIR = '../data/'
MODEL_DIR = '../models/'
INPUT_FILE_X = 'X_test.csv'
INPUT_FILE_y = 'y_test.csv'
AE_MODEL_PATH = os.path.join(MODEL_DIR, 'ldrae.pth')
SVM_MODEL_PATH = os.path.join(MODEL_DIR, 'best_svm_fixed_ae.joblib')

INPUT_DIM = 119
HIDDEN_DIM_1 = 64
HIDDEN_DIM_2 = 32

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
    
    y_test = 1 - y_test_df.iloc[:, 0].values # Flip to 1=Anomaly
    
    X_test_tensor = torch.tensor(X_test_df.values, dtype=torch.float32)
    
    print("Loading AE model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_model = Autoencoder().to(device)
    ae_model.load_state_dict(torch.load(AE_MODEL_PATH, map_location=device))
    ae_model.eval()
    
    print("Extracting latent features...")
    latent_vectors = []
    batch_size = 256
    dataset = TensorDataset(X_test_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for data in loader:
            inputs = data[0].to(device)
            latent = ae_model.encoder(inputs)
            latent_vectors.append(latent.cpu().numpy())
            
    X_test_latent = np.concatenate(latent_vectors, axis=0)
    
    print("Loading Best SVM model...")
    svm_model = joblib.load(SVM_MODEL_PATH)
    
    print("Predicting...")
    y_pred_raw = svm_model.predict(X_test_latent)
    y_pred = 1 - y_pred_raw # Flip predictions
    
    print("\n--- Evaluation Results (Optimized Linear SVM on Fixed AE) ---")
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
