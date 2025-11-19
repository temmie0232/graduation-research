import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# --- Configuration ---
DATA_DIR = '../data/'
MODEL_DIR = '../models/'
INPUT_FILE_X = 'X_train.csv'
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'ldrae.pth')

# Hyperparameters
INPUT_DIM = 119
HIDDEN_DIM_1 = 64
HIDDEN_DIM_2 = 32
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
RECONSTRUCTION_PERCENTILE = 0.5346 # Use bottom 50% of samples with lowest reconstruction error for training

# --- 1. Load Data ---
print("Loading all training data (normal and anomaly)...")
X_train_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_X))

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_df.values, dtype=torch.float32)

# --- 2. Create DataLoader ---
# For LDR-AE, we use the full dataset, not just normal data.
# The target is the same as the input.
train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Training with {len(X_train_df)} total samples (normal and anomaly).")

# --- 3. Define the Autoencoder Model ---
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

# --- 4. Training Loop (Learning Discriminative Reconstruction) ---
model = Autoencoder()
criterion = nn.MSELoss(reduction='none') # Use 'none' to get per-sample loss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nStarting LDR-AE training...")
for epoch in range(EPOCHS):
    epoch_loss = 0
    processed_samples = 0
    for data in train_loader:
        inputs, _ = data
        
        # 1. Get reconstructions
        outputs = model(inputs)
        
        # 2. Calculate reconstruction error for each sample in the batch
        reconstruction_errors = torch.mean((outputs - inputs) ** 2, dim=1)
        
        # 3. Discriminative Labeling: Find the threshold for "normal" samples
        # Samples with error below the percentile are considered "normal" for this batch
        threshold = torch.quantile(reconstruction_errors, RECONSTRUCTION_PERCENTILE)
        
        # 4. Reconstruction Learning: Select only the "normal" samples
        normal_mask = reconstruction_errors <= threshold
        
        # If no samples are considered normal, skip this batch
        if normal_mask.sum() == 0:
            continue
            
        selected_inputs = inputs[normal_mask]
        selected_outputs = outputs[normal_mask]
        
        # 5. Calculate loss only on the selected "normal" samples
        loss = criterion(selected_outputs, selected_inputs).mean() # Now, average the loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * selected_inputs.size(0)
        processed_samples += selected_inputs.size(0)
    
    avg_epoch_loss = epoch_loss / processed_samples if processed_samples > 0 else 0
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_epoch_loss:.6f}')

# --- 5. Save the Model ---
print("\nTraining complete. Saving model...")
os.makedirs(MODEL_DIR, exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
