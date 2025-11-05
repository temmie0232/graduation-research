import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define the column names for the NSL-KDD dataset
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'attack', 'last_flag'
]

# Load the training and testing datasets
try:
    train_df = pd.read_csv('../data/KDDTrain+.txt', header=None, names=columns)
    test_df = pd.read_csv('../data/KDDTest+.txt', header=None, names=columns)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure the dataset files 'KDDTrain+.txt' and 'KDDTest+.txt' are in the 'data' directory.")
    exit()

print("Original training data shape:", train_df.shape)
print("Original testing data shape:", test_df.shape)

# --- Preprocessing Steps ---

# 1. Combine train and test for consistent encoding
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# 2. Identify categorical and numerical columns
categorical_cols = ['protocol_type', 'service', 'flag']
numerical_cols = combined_df.select_dtypes(include=['number']).columns.tolist()
# Remove the label and irrelevant columns from numerical cols
for col in ['attack', 'last_flag']:
    if col in numerical_cols:
        numerical_cols.remove(col)

# 3. One-Hot Encode categorical features
combined_df = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True)

# 4. Separate back into training and testing sets
train_len = len(train_df)
X_train = combined_df.iloc[:train_len].drop(columns=['attack', 'last_flag'])
X_test = combined_df.iloc[train_len:].drop(columns=['attack', 'last_flag'])
y_train = train_df['attack']
y_test = test_df['attack']

# Align columns after one-hot encoding - crucial for consistency
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 5. Convert labels to binary (1 for normal, 0 for attack)
y_train_binary = y_train.apply(lambda x: 1 if x == 'normal' else 0)
y_test_binary = y_test.apply(lambda x: 1 if x == 'normal' else 0)

# 6. Scale numerical features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# 7. Save the preprocessed data
output_dir = '../data/'
X_train_scaled_df.to_csv(output_dir + 'X_train.csv', index=False)
y_train_binary.to_csv(output_dir + 'y_train.csv', index=False, header=True)
X_test_scaled_df.to_csv(output_dir + 'X_test.csv', index=False)
y_test_binary.to_csv(output_dir + 'y_test.csv', index=False, header=True)

print("\nPreprocessing complete.")
print("Preprocessed training data shape:", X_train_scaled_df.shape)
print("Preprocessed testing data shape:", X_test_scaled_df.shape)
print(f"Saved preprocessed files to '{output_dir}'")