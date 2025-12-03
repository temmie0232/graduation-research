import pandas as pd
import os
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

# --- Configuration ---
DATA_DIR = '../data/'
MODEL_DIR = '../models/'
INPUT_FILE_X = 'X_train.csv'
INPUT_FILE_y = 'y_train.csv'
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'raw_svm.joblib')

# --- 1. Load Data ---
print("Loading data...")
X_train_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_X))
y_train_df = pd.read_csv(os.path.join(DATA_DIR, INPUT_FILE_y))

# Use all features
X_train = X_train_df.values
y_train = y_train_df.iloc[:, 0].values # Assuming label is in the first column

print(f"Input shape: {X_train.shape}")

# --- 2. Train SVM ---
print("Training Raw SVM classifier...")
# Use LinearSVC for speed, wrapped in CalibratedClassifierCV for probability estimation
# We add StandardScaler to be safe, even if data was MinMax scaled, centering is good for SVM.
linear_svc = LinearSVC(dual=False, random_state=42)
svm_clf = make_pipeline(StandardScaler(), CalibratedClassifierCV(linear_svc))
svm_clf.fit(X_train, y_train)

# --- 3. Save SVM Model ---
print("Saving SVM model...")
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(svm_clf, MODEL_SAVE_PATH)
print(f"SVM model saved to {MODEL_SAVE_PATH}")
