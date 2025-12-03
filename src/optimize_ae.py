import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import os
import numpy as np

# --- Configuration ---
DATA_DIR = '../data/'
MODEL_DIR = '../models/'
INPUT_FILE_X = 'X_train.csv'
INPUT_FILE_y = 'y_train.csv'
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_ae_optuna.pth')

INPUT_DIM = 119
EPOCHS = 10 # Keep low for optimization speed, or use pruning

def load_data():
    X_train_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_X))
    y_train_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_y))
    
    # Filter for normal data
    normal_indices = y_train_df[y_train_df.iloc[:, 0] == 1].index
    X_train_normal = X_train_df.iloc[normal_indices]
    
    return torch.tensor(X_train_normal.values, dtype=torch.float32)

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

def objective(trial):
    # Hyperparameters to search
    hidden_dim_1 = trial.suggest_categorical('hidden_dim_1', [64, 128, 256])
    hidden_dim_2 = trial.suggest_categorical('hidden_dim_2', [16, 24, 32])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # Data Loading
    X_train_tensor = load_data()
    
    # Split into train/val
    val_size = int(0.2 * len(X_train_tensor))
    train_size = len(X_train_tensor) - val_size
    train_dataset, val_dataset = random_split(TensorDataset(X_train_tensor, X_train_tensor), [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(INPUT_DIM, hidden_dim_1, hidden_dim_2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            inputs, _ = batch
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, _ = batch
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Pruning (optional)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_loss

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20) # Run 20 trials
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Retrain best model on full data and save
    print("\nRetraining best model on full data...")
    best_params = trial.params
    X_train_tensor = load_data()
    full_loader = DataLoader(TensorDataset(X_train_tensor, X_train_tensor), batch_size=best_params['batch_size'], shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = Autoencoder(INPUT_DIM, best_params['hidden_dim_1'], best_params['hidden_dim_2']).to(device)
    optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
    criterion = nn.MSELoss()
    
    for epoch in range(EPOCHS): # Could increase epochs for final training
        best_model.train()
        for batch in full_loader:
            inputs, _ = batch
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = best_model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
    torch.save(best_model.state_dict(), BEST_MODEL_PATH)
    print(f"Best model saved to {BEST_MODEL_PATH}")
    
    # Save params for next step
    import json
    with open(os.path.join(MODEL_DIR, 'best_ae_params.json'), 'w') as f:
        json.dump(best_params, f)
