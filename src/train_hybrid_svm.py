import pandas as pd
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import os
import numpy as np

# --- Configuration ---
DATA_DIR = '../data/'
MODEL_DIR = '../models/'
INPUT_FILE_X = 'X_train.csv'
INPUT_FILE_y = 'y_train.csv'
AE_MODEL_PATH = os.path.join(MODEL_DIR, 'ldrae.pth')
SVM_MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'svm_classifier.joblib')

# Hyperparameters from AE training
INPUT_DIM = 119
HIDDEN_DIM_1 = 64
HIDDEN_DIM_2 = 32 # Latent space

# --- 1. Define the Autoencoder Model (must match the trained model) ---
# We only need the encoder part to get the latent variables.
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
        # We don't need the decoder for this script
        return x

# --- 2. Load Data and AE Model ---
print("Loading data and LDR-AE model...")
X_train_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_X))
y_train_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_y))
y_train_labels = y_train_df.iloc[:, 0].values

X_train_tensor = torch.tensor(X_train_df.values, dtype=torch.float32)

ae_model = Autoencoder()
ae_model.load_state_dict(torch.load(AE_MODEL_PATH))
ae_model.eval() # Set to evaluation mode

# --- 3. Extract Latent Variables ---
print("Extracting latent variables from the training data...")
with torch.no_grad():
    latent_variables = ae_model(X_train_tensor).numpy()

print(f"Extracted latent variables with shape: {latent_variables.shape}")

# --- 4. Scale Latent Variables ---
# SVMs are sensitive to feature scaling.
print("Scaling latent variables...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(latent_variables)

# Save the scaler to use it on test data later
scaler_save_path = os.path.join(MODEL_DIR, 'svm_scaler.joblib')
joblib.dump(scaler, scaler_save_path)
print(f"Scaler saved to {scaler_save_path}")

# --- 5. Train the SVM Classifier ---
print("\nTraining SVM classifier...")
# Using a subset for faster training, as SVM can be slow on large datasets.
# This is a common practice. Let's take 30,000 random samples.
sample_size = 30000
if len(X_train_scaled) > sample_size:
    print(f"Using a random subset of {sample_size} samples for SVM training.")
    indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
    X_train_subset = X_train_scaled[indices]
    y_train_subset = y_train_labels[indices]
else:
    X_train_subset = X_train_scaled
    y_train_subset = y_train_labels

# Initialize and train SVM
# Using class_weight='balanced' is important for imbalanced datasets like NSL-KDD
svm_classifier = SVC(kernel='rbf', gamma='scale', class_weight='balanced', probability=True, verbose=True)
svm_classifier.fit(X_train_subset, y_train_subset)

# --- 6. Save the SVM Model ---
print("\nTraining complete. Saving SVM model...")
joblib.dump(svm_classifier, SVM_MODEL_SAVE_PATH)
print(f"SVM model saved to {SVM_MODEL_SAVE_PATH}")
