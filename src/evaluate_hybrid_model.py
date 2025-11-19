import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import joblib
import os

# --- Configuration ---
DATA_DIR = '../data/'
MODEL_DIR = '../models/'
INPUT_FILE_X = 'X_test.csv'
INPUT_FILE_y = 'y_test.csv'
AE_MODEL_PATH = os.path.join(MODEL_DIR, 'ldrae.pth')
SVM_MODEL_PATH = os.path.join(MODEL_DIR, 'svm_classifier.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'svm_scaler.joblib')

# Hyperparameters from AE training
INPUT_DIM = 119
HIDDEN_DIM_1 = 64
HIDDEN_DIM_2 = 32 # Latent space

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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# --- 2. Load All Models and Data ---
print("Loading data and all models (AE, SVM, Scaler)...")
X_test_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_X))
y_test_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_y))
y_test_true = y_test_df.iloc[:, 0].values

X_test_tensor = torch.tensor(X_test_df.values, dtype=torch.float32)

# Load AE
ae_model = Autoencoder()
ae_model.load_state_dict(torch.load(AE_MODEL_PATH))
ae_model.eval()

# Load SVM and Scaler
svm_classifier = joblib.load(SVM_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --- 3. Calculate Both Anomaly Scores ---
print("Calculating anomaly scores from both AE and SVM...")
with torch.no_grad():
    reconstructions, latent_variables = ae_model(X_test_tensor)
    
    # Score A: Reconstruction Error (higher is more anomalous)
    recon_errors = np.mean((X_test_tensor.numpy() - reconstructions.numpy())**2, axis=1)
    
    # Score B: SVM Anomaly Probability (higher is more anomalous)
    scaled_latent = scaler.transform(latent_variables.numpy())
    # Get probability of being class '0' (anomaly)
    svm_anomaly_prob = svm_classifier.predict_proba(scaled_latent)[:, 0]

# --- 4. Normalize and Combine Scores ---
print("Normalizing and combining scores...")
# Reshape for MinMaxScaler
recon_errors_reshaped = recon_errors.reshape(-1, 1)
svm_prob_reshaped = svm_anomaly_prob.reshape(-1, 1)

# Normalize both scores to be in [0, 1]
score_scaler = MinMaxScaler()
norm_recon_errors = score_scaler.fit_transform(recon_errors_reshaped).flatten()
norm_svm_prob = score_scaler.fit_transform(svm_prob_reshaped).flatten()

# Combine scores (simple average)
hybrid_score = (norm_recon_errors + norm_svm_prob) / 2.0

errors_df = pd.DataFrame({
    'recon_error': recon_errors,
    'svm_prob': svm_anomaly_prob,
    'hybrid_score': hybrid_score,
    'true_class': y_test_true
})

# --- 5. Set Threshold and Classify using Hybrid Score ---
print("Finding optimal threshold for hybrid score...")
fpr, tpr, thresholds = roc_curve(errors_df.true_class, errors_df.hybrid_score, pos_label=0)
j_scores = tpr - fpr
best_threshold = thresholds[np.argmax(j_scores)]

print(f"\nChosen Hybrid Threshold (Youden's J): {best_threshold:.6f}")

y_pred = [1 if e < best_threshold else 0 for e in errors_df.hybrid_score]

# --- 6. Evaluate Performance ---
print("\nEvaluating Hybrid Model performance...")
accuracy = accuracy_score(y_test_true, y_pred)
cm = confusion_matrix(y_test_true, y_pred)
report = classification_report(y_test_true, y_pred, target_names=['Anomaly (0)', 'Normal (1)'])

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

# Save confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Anomaly', 'Normal'], yticklabels=['Anomaly', 'Normal'])
plt.title('Confusion Matrix (Hybrid Model)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(os.path.join(MODEL_DIR, 'hybrid_confusion_matrix.png'))
print(f"\nSaved confusion matrix plot to '{MODEL_DIR}hybrid_confusion_matrix.png'")
