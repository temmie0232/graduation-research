import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import os

# --- Configuration ---
DATA_DIR = '../data/'
MODEL_DIR = '../models/'
INPUT_FILE_X = 'X_test.csv'
INPUT_FILE_y = 'y_test.csv'
MODEL_PATH = os.path.join(MODEL_DIR, 'ldrae.pth')

# Hyperparameters from training script
INPUT_DIM = 119
HIDDEN_DIM_1 = 64
HIDDEN_DIM_2 = 32 # Latent space

# --- Define the Autoencoder Model (must match the trained model) ---
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

# --- 1. Load Data and Model ---
print("Loading data and LDR-AE model...")
X_test_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_X))
y_test_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_y))
y_test_true = y_test_df.iloc[:, 0].values

X_test_tensor = torch.tensor(X_test_df.values, dtype=torch.float32)

model = Autoencoder()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval() # Set the model to evaluation mode

# --- 2. Calculate Reconstruction Errors ---
print("Calculating reconstruction errors...")
with torch.no_grad():
    reconstructions = model(X_test_tensor)
    mse = np.mean((X_test_tensor.numpy() - reconstructions.numpy())**2, axis=1)

errors_df = pd.DataFrame({'recon_error': mse, 'true_class': y_test_true})

# --- 3. Visualize Error Distribution ---
print("Visualizing error distribution...")
plt.figure(figsize=(12, 6))
sns.histplot(data=errors_df, x='recon_error', hue='true_class', kde=True, bins=50, palette={0: 'red', 1: 'blue'})
plt.title('Distribution of Reconstruction Errors for Normal (1) and Anomaly (0) Data (LDR-AE)')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.savefig(os.path.join(MODEL_DIR, 'ldrae_reconstruction_error_distribution.png'))
print(f"Saved error distribution plot to '{MODEL_DIR}ldrae_reconstruction_error_distribution.png'")

# --- 4. Set Threshold and Classify (using Youden's J-statistic) ---
# Find the optimal threshold using the ROC curve
# Note: roc_curve expects scores where higher is more likely to be positive.
# Our errors are such that lower is normal (positive class, 1), so we use (1 - true_class)
# and the negative of the error to flip the relationship.
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test_true, errors_df.recon_error, pos_label=0) # type: ignore

# Calculate Youden's J-statistic
j_scores = tpr - fpr
j_ordered = sorted(zip(j_scores, thresholds))
# best_threshold = thresholds[np.argmax(j_scores)]
best_threshold = j_ordered[-1][1]

print(f"\nChosen Threshold (Youden's J): {best_threshold:.6f}")

y_pred = [1 if e < best_threshold else 0 for e in errors_df.recon_error]

# --- 5. Evaluate Performance ---
print("\nEvaluating performance...")
accuracy = accuracy_score(y_test_true, y_pred) # type: ignore
cm = confusion_matrix(y_test_true, y_pred) # type: ignore
report = classification_report(y_test_true, y_pred, target_names=['Anomaly (0)', 'Normal (1)']) # type: ignore

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

# Save confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Anomaly', 'Normal'], yticklabels=['Anomaly', 'Normal'])
plt.title('Confusion Matrix (LDR-AE)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(os.path.join(MODEL_DIR, 'ldrae_confusion_matrix.png'))
print(f"\nSaved confusion matrix plot to '{MODEL_DIR}ldrae_confusion_matrix.png'")
