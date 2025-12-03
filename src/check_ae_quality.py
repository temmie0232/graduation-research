import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import json
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

# --- Configuration ---
DATA_DIR = '../data/'
MODEL_DIR = '../models/'
INPUT_FILE_X_TRAIN = 'X_train.csv'
INPUT_FILE_y_TRAIN = 'y_train.csv'
INPUT_FILE_X_TEST = 'X_test.csv'
INPUT_FILE_y_TEST = 'y_test.csv'
AE_PARAMS_PATH = os.path.join(MODEL_DIR, 'best_ae_params.json')
AE_MODEL_PATH = os.path.join(MODEL_DIR, 'best_ae_optuna.pth')

INPUT_DIM = 119

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

def get_latent(X_df, ae_model, device):
    X_tensor = torch.tensor(X_df.values, dtype=torch.float32)
    latent_vectors = []
    batch_size = 256
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for data in loader:
            inputs = data[0].to(device)
            latent = ae_model.encoder(inputs)
            latent_vectors.append(latent.cpu().numpy())
    return np.concatenate(latent_vectors, axis=0)

if __name__ == "__main__":
    print("Loading data...")
    X_train_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_X_TRAIN))
    y_train_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_y_TRAIN))
    X_test_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_X_TEST))
    y_test_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_y_TEST))
    
    y_train = y_train_df.iloc[:, 0].values
    y_test = 1 - y_test_df.iloc[:, 0].values # Flip for evaluation (1=Anomaly)
    
    print("Loading AE...")
    with open(AE_PARAMS_PATH, 'r') as f:
        ae_params = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_model = Autoencoder(INPUT_DIM, ae_params['hidden_dim_1'], ae_params['hidden_dim_2']).to(device)
    ae_model.load_state_dict(torch.load(AE_MODEL_PATH, map_location=device))
    ae_model.eval()
    
    print("Extracting features...")
    X_train_latent = get_latent(X_train_df, ae_model, device)
    X_test_latent = get_latent(X_test_df, ae_model, device)
    
    print("Training Simple LinearSVC...")
    clf = make_pipeline(StandardScaler(), LinearSVC(dual=False, random_state=42))
    clf.fit(X_train_latent, y_train)
    
    print("Evaluating...")
    y_pred_raw = clf.predict(X_test_latent)
    y_pred = 1 - y_pred_raw # Flip predictions (SVM trained on 1=Normal)
    
    f1 = f1_score(y_test, y_pred)
    print(f"Simple LinearSVC F1 Score: {f1:.4f}")
