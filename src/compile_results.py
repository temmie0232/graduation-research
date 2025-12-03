import os
import pandas as pd

MODEL_DIR = '../models/'
RAW_SVM_RESULTS = os.path.join(MODEL_DIR, 'raw_svm_results.txt')
AE_DIMS_RESULTS = os.path.join(MODEL_DIR, 'ae_dims_comparison.txt')
SVM_KERNEL_RESULTS = os.path.join(MODEL_DIR, 'svm_kernel_comparison.txt')

print("# Milestone 3.5 Experiment Results\n")

# 1. Raw SVM
print("## 1. Baseline 2: Raw SVM (No AE)")
if os.path.exists(RAW_SVM_RESULTS):
    with open(RAW_SVM_RESULTS, 'r') as f:
        print(f.read())
else:
    print("Results not found.")

# 2. AE Dimensions
print("\n## 2. AE Dimension Comparison")
if os.path.exists(AE_DIMS_RESULTS):
    df = pd.read_csv(AE_DIMS_RESULTS)
    print(df.to_string(index=False))
else:
    print("Results not found.")

# 3. SVM Kernels
print("\n## 3. SVM Kernel Comparison")
if os.path.exists(SVM_KERNEL_RESULTS):
    df = pd.read_csv(SVM_KERNEL_RESULTS)
    print(df.to_string(index=False))
else:
    print("Results not found.")
