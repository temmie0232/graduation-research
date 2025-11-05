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
INPUT_FILE_y = 'y_train.csv'
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'baseline_ae.pth')

# Hyperparameters
INPUT_DIM = 119 # From the shape of preprocessed data
HIDDEN_DIM_1 = 64
HIDDEN_DIM_2 = 32 # Latent space
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# --- 1. Load Data ---
print("Loading data...")
X_train_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_X))
y_train_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_y))

# --- 2. Filter for Normal Data ---
print("Filtering for normal data...")
normal_indices = y_train_df[y_train_df.iloc[:, 0] == 1].index
X_train_normal = X_train_df.iloc[normal_indices]

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_normal.values, dtype=torch.float32)

# --- 3. Create DataLoader ---
train_dataset = TensorDataset(X_train_tensor, X_train_tensor) # Input and target are the same for AE
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Training with {len(X_train_normal)} normal samples.")

# --- 4. Define the Autoencoder Model ---
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
            nn.Sigmoid()  # Use Sigmoid as data is scaled to [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- 5. Training Loop ---
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nStarting training...")
for epoch in range(EPOCHS):
    epoch_loss = 0
    for data in train_loader:
        inputs, _ = data
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(train_loader):.6f}')

# --- 6. Save the Model ---
print("\nTraining complete. Saving model...")
os.makedirs(MODEL_DIR, exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
