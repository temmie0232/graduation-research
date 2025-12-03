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
AE_PARAMS_PATH = os.path.join(MODEL_DIR, 'best_ae_params.json')
AE_MODEL_PATH = os.path.join(MODEL_DIR, 'best_ae_optuna.pth')
SVM_MODEL_PATH = os.path.join(MODEL_DIR, 'best_svm_optuna.joblib')

INPUT_DIM = 119

# --- AE Definition ---
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

if __name__ == "__main__":
    print("Loading test data...")
    X_test_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_X))
    y_test_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_y))
    
    # Preprocess labels: 1=Normal, 0=Anomaly in dataset -> Convert to 1=Anomaly, 0=Normal for standard metrics
    # Wait, check preprocess.py or previous logs.
    # In Work_Log.md: "y_test = 1 - y_test" was used to fix evaluation.
    # Original: 1=Normal, 0=Anomaly.
    # Target: 1=Anomaly, 0=Normal.
    y_test = 1 - y_test_df.iloc[:, 0].values
    
    X_test_tensor = torch.tensor(X_test_df.values, dtype=torch.float32)
    
    print("Loading Best AE model...")
    with open(AE_PARAMS_PATH, 'r') as f:
        ae_params = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_model = Autoencoder(INPUT_DIM, ae_params['hidden_dim_1'], ae_params['hidden_dim_2']).to(device)
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
    # SVM was trained on labels where 1=Normal?
    # Wait, in optimize_svm.py: y_train = y_train_df.iloc[:, 0].values
    # y_train_df comes from preprocess.py, where 1=Normal, 0=Anomaly.
    # So SVM predicts 1 for Normal, 0 for Anomaly.
    
    y_pred_raw = svm_model.predict(X_test_latent)
    
    # Convert predictions to 1=Anomaly, 0=Normal
    y_pred = 1 - y_pred_raw
    
    print("\n--- Evaluation Results (Optimized Model) ---")
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
    with open(os.path.join(MODEL_DIR, 'optimized_results.txt'), 'w') as f:
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
