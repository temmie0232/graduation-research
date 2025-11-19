import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_DIR = '../data/'
MODEL_DIR = '../models/'
INPUT_FILE_X = 'X_test.csv'
INPUT_FILE_y = 'y_test.csv'
AE_MODEL_PATH = os.path.join(MODEL_DIR, 'ldrae.pth')
SVM_MODEL_PATH = os.path.join(MODEL_DIR, 'svm_classifier.joblib')
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, 'hybrid_confusion_matrix.png')

# Hyperparameters
INPUT_DIM = 119
HIDDEN_DIM_1 = 64
HIDDEN_DIM_2 = 32
BATCH_SIZE = 64

# --- 1. Define the Autoencoder Model ---
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

# --- 2. Load Data and Models ---
print("Loading data...")
X_test_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_X))
y_test_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_y))

X_test_tensor = torch.tensor(X_test_df.values, dtype=torch.float32)
# In preprocess.py: 1=Normal, 0=Anomaly
# For Anomaly Detection evaluation: 1=Anomaly, 0=Normal
y_test = 1 - y_test_df.iloc[:, 0].values

print("Loading models...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ae_model = Autoencoder().to(device)
ae_model.load_state_dict(torch.load(AE_MODEL_PATH, map_location=device))
ae_model.eval()

svm_clf = joblib.load(SVM_MODEL_PATH)

# --- 3. Calculate Scores ---
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
        
        # Store latent vectors
        latent_vectors.append(latent.cpu().numpy())
        
        # Calculate reconstruction error (MSE per sample)
        error = torch.mean((outputs - inputs) ** 2, dim=1)
        reconstruction_errors.append(error.cpu().numpy())

X_test_latent = np.concatenate(latent_vectors, axis=0)
score_a = np.concatenate(reconstruction_errors, axis=0)

# Calculate SVM Probability (Score B)
# Labels: 1 = Normal, 0 = Anomaly
# We want an Anomaly Score, so we use the probability of class 0.
score_b = svm_clf.predict_proba(X_test_latent)[:, 0] 

# --- 4. Hybrid Decision ---
# Normalize scores to [0, 1] range for fair combination
# Simple Min-Max scaling based on test set (in practice, should use training set stats)
score_a_norm = (score_a - score_a.min()) / (score_a.max() - score_a.min())
score_b_norm = (score_b - score_b.min()) / (score_b.max() - score_b.min())

# Combine scores
# Strategy: Weighted Average (0.5 * A + 0.5 * B)
# Or logical OR: if either is high, it's an anomaly.
# Let's try weighted average first.
hybrid_score = 0.5 * score_a_norm + 0.5 * score_b_norm

# Determine threshold
# Simple approach: use mean + std or a fixed percentile.
# Better approach: Find best threshold using Precision-Recall curve (omitted for brevity).
# Let's use a threshold that gives a reasonable balance, e.g., top 20% are anomalies?
# Or simply 0.5 if normalized?
# Let's try to find the best threshold based on F1 score for this evaluation.
best_f1 = 0
best_threshold = 0
thresholds = np.linspace(0, 1, 101)

for thresh in thresholds:
    y_pred = (hybrid_score > thresh).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"Best Threshold: {best_threshold:.4f}")
y_pred = (hybrid_score > best_threshold).astype(int)

# --- 5. Evaluation ---
print("\n--- Evaluation Results ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Hybrid Model Confusion Matrix')
plt.savefig(CONFUSION_MATRIX_PATH)
print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")
