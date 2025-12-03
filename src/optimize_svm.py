import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import numpy as np
import json
import joblib
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.kernel_approximation import Nystroem
from sklearn.calibration import CalibratedClassifierCV

# --- Configuration ---
DATA_DIR = '../data/'
MODEL_DIR = '../models/'
INPUT_FILE_X = 'X_train.csv'
INPUT_FILE_y = 'y_train.csv'
AE_PARAMS_PATH = os.path.join(MODEL_DIR, 'best_ae_params.json')
AE_MODEL_PATH = os.path.join(MODEL_DIR, 'best_ae_optuna.pth')
BEST_SVM_PATH = os.path.join(MODEL_DIR, 'best_svm_optuna.joblib')

INPUT_DIM = 119

# --- AE Definition (Must match optimize_ae.py) ---
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

def load_data_and_extract_features():
    # Load Params
    with open(AE_PARAMS_PATH, 'r') as f:
        ae_params = json.load(f)
    print(f"DEBUG: Loaded AE params: {ae_params}")
    
    # Load Data
    X_train_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_X))
    y_train_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_y))
    
    X_train_tensor = torch.tensor(X_train_df.values, dtype=torch.float32)
    y_train = y_train_df.iloc[:, 0].values
    
    # Load AE Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_model = Autoencoder(INPUT_DIM, ae_params['hidden_dim_1'], ae_params['hidden_dim_2']).to(device)
    ae_model.load_state_dict(torch.load(AE_MODEL_PATH, map_location=device))
    ae_model.eval()
    
    # Extract Features
    latent_vectors = []
    batch_size = 256
    dataset = TensorDataset(X_train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for data in loader:
            inputs = data[0].to(device)
            latent = ae_model.encoder(inputs)
            latent_vectors.append(latent.cpu().numpy())
            
    X_latent = np.concatenate(latent_vectors, axis=0)
    print(f"DEBUG: Extracted latent features shape: {X_latent.shape}")
    return X_latent, y_train

X_latent, y_train = None, None

def objective(trial):
    global X_latent, y_train
    if X_latent is None:
        X_latent, y_train = load_data_and_extract_features()
        
    # Hyperparameters
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'nystroem'])
    C = trial.suggest_float('C', 0.1, 100, log=True)
    
    if kernel == 'linear':
        clf = LinearSVC(dual=False, C=C, random_state=42)
        model = make_pipeline(StandardScaler(), clf)
        
    elif kernel == 'nystroem':
        gamma = trial.suggest_categorical('gamma_nystroem', [0.001, 0.01, 0.1, 1.0])
        n_components = trial.suggest_int('n_components', 50, 500)
        feature_map = Nystroem(gamma=gamma, n_components=n_components, random_state=42)
        clf = LinearSVC(dual=False, C=C, random_state=42)
        model = make_pipeline(StandardScaler(), feature_map, clf)
        
    elif kernel == 'rbf':
        # Use a subset for RBF speed during optimization if needed, but let's try full first or small subset
        # RBF is slow on large data.
        gamma = trial.suggest_categorical('gamma_rbf', ['scale', 'auto', 0.01, 0.1])
        clf = SVC(kernel='rbf', C=C, gamma=gamma, cache_size=1000, random_state=42)
        model = make_pipeline(StandardScaler(), clf)

    # Cross Validation (3-fold to save time)
    # Use F1 score
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # For RBF, maybe subsample data for optimization speed?
    # Let's subsample to 2000 samples for optimization loop to be fast
    indices = np.random.choice(len(X_latent), 2000, replace=False)
    X_sub = X_latent[indices]
    y_sub = y_train[indices]
    
    scores = cross_val_score(model, X_sub, y_sub, cv=cv, scoring='f1', n_jobs=-1)
    return scores.mean()

if __name__ == "__main__":
    # Ensure AE model exists
    if not os.path.exists(AE_PARAMS_PATH):
        print("AE params not found. Run optimize_ae.py first.")
        exit(1)
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Train best model on full data
    print("\nTraining best model on full data...")
    X_latent, y_train = load_data_and_extract_features()
    
    best_params = trial.params
    kernel = best_params['kernel']
    C = best_params['C']
    
    if kernel == 'linear':
        clf = LinearSVC(dual=False, C=C, random_state=42)
        # Wrap in CalibratedClassifierCV for probability
        final_clf = CalibratedClassifierCV(clf)
        model = make_pipeline(StandardScaler(), final_clf)
        
    elif kernel == 'nystroem':
        gamma = best_params['gamma_nystroem']
        n_components = best_params['n_components']
        feature_map = Nystroem(gamma=gamma, n_components=n_components, random_state=42)
        clf = LinearSVC(dual=False, C=C, random_state=42)
        final_clf = CalibratedClassifierCV(clf)
        model = make_pipeline(StandardScaler(), feature_map, final_clf)
        
    elif kernel == 'rbf':
        gamma = best_params['gamma_rbf']
        clf = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, cache_size=1000, random_state=42)
        model = make_pipeline(StandardScaler(), clf)
        
    model.fit(X_latent, y_train)
    
    joblib.dump(model, BEST_SVM_PATH)
    print(f"Best SVM model saved to {BEST_SVM_PATH}")
