import pandas as pd
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
MODEL_PATH = os.path.join(MODEL_DIR, 'raw_svm.joblib')
RESULTS_PATH = os.path.join(MODEL_DIR, 'raw_svm_results.txt')
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, 'raw_svm_confusion_matrix.png')

# --- 1. Load Data and Model ---
print("Loading data...")
X_test_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_X))
y_test_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_y))

X_test = X_test_df.values
# In preprocess.py: 1=Normal, 0=Anomaly
# For Anomaly Detection evaluation: 1=Anomaly, 0=Normal
y_test = 1 - y_test_df.iloc[:, 0].values

print(f"Loading Raw SVM model from {MODEL_PATH}...")
svm_clf = joblib.load(MODEL_PATH)

# --- 2. Predict ---
print("Predicting...")
# Predict probabilities for class 0 (Anomaly)
# The model was trained on original labels (1=Normal, 0=Anomaly)
# So class 0 is Anomaly.
probs = svm_clf.predict_proba(X_test)
# probs[:, 0] is probability of class 0 (Anomaly)
anomaly_scores = probs[:, 0]

# Predict labels
# If probability of anomaly > 0.5, then it's an anomaly
y_pred = (anomaly_scores > 0.5).astype(int)

# --- 3. Evaluation ---
print("\n--- Evaluation Results (Raw SVM) ---")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Save results to file
with open(RESULTS_PATH, 'w') as f:
    f.write("--- Evaluation Results (Raw SVM) ---\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Raw SVM Confusion Matrix')
plt.savefig(CONFUSION_MATRIX_PATH)
print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")
