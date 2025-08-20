import numpy as np
import pandas as pd
import os

files = {
    "v1": "Data/Processed/cleaned_bridge_dataset_V1.csv",
    "v2": "Data/Processed/featured_dataset_V2.csv",
    "v3": "Data/Processed/V3_resampled_dataset.csv",
    "v3_test": "Data/Processed/test_data.csv"
}

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def to_numpy_if_df(x):
    return x.values if isinstance(x, pd.DataFrame) else np.array(x)

def prepare_X_as_3d(X):
    X = to_numpy_if_df(X)
    if X.ndim == 1:   # (n,) → (n,1,1)
        X = X.reshape(-1, 1, 1)
    elif X.ndim == 2: # (n,features) → (n,1,features)
        X = X.reshape(X.shape[0], 1, X.shape[1])
    elif X.ndim != 3:
        raise ValueError(f"Unsupported X.ndim = {X.ndim}")
    return X.astype(np.float32)

def prepare_y(y):
    """Ensure y is 1D int64 array (targets already integers)."""
    y = to_numpy_if_df(y).squeeze()
    if y.ndim != 1:
        y = y.reshape(-1)
    return y.astype(np.int64), None   # no mapping needed

# --- load ---
print("Loading files (will raise if a path is incorrect):")
df_v1 = load_csv(files["v1"])
df_v2 = load_csv(files["v2"])
df_v3 = load_csv(files["v3"])
df_test_v3 = load_csv(files["v3_test"])

# --- convert to numpy arrays ---
X1_raw = df_v1.drop(["damage_class", "structural_condition", "date", "time"], axis=1, errors='ignore')
y1_raw = df_v1.get("structural_condition")

X2_raw = df_v2.drop(["damage_class", "structural_condition", "date", "time"], axis=1, errors='ignore')
y2_raw = df_v2.get("structural_condition")

X3_test_raw = df_test_v3.drop(["damage_class", "structural_condition", "date", "time"], axis=1, errors='ignore')
y3_test_raw = df_test_v3.get("structural_condition")

X3_train_raw = df_v3.drop(["damage_class", "structural_condition", "date", "time"], axis=1, errors='ignore')
y3_train_raw = df_v3.get("structural_condition")

# Prepare shapes
X1 = prepare_X_as_3d(X1_raw)
X2 = prepare_X_as_3d(X2_raw)
X3_train = prepare_X_as_3d(X3_train_raw)
X3_test = prepare_X_as_3d(X3_test_raw)

y1, _ = prepare_y(y1_raw) if y1_raw is not None else (None, None)
y2, _ = prepare_y(y2_raw) if y2_raw is not None else (None, None)
y3_test, _ = prepare_y(y3_test_raw) if y3_test_raw is not None else (None, None)
y3_train, _ = prepare_y(y3_train_raw)

# Save arrays
np.savez_compressed("Data/Processed/Training_Prepared_Data/V1_dataset.npz", X=X1 , y=y1)
np.savez_compressed("Data/Processed/Training_Prepared_Data/V2_dataset.npz", X=X2 , y=y2)
np.savez_compressed("Data/Processed/Training_Prepared_Data/V3_resampled_dataset.npz", X=X3_train , y=y3_train)
np.savez_compressed("Data/Processed/Testing_Prepared_Data/V3_test_resampled.npz", X=X3_test , y=y3_test)

# --- After prepare ---
print("\n--- After prepare ---")
print("X1 shape (N, seq_len, channels):", X1.shape, "dtype:", X1.dtype)
print("X2 shape (N, seq_len, channels):", X2.shape, "dtype:", X2.dtype)
print("X3_train shape (N, seq_len, channels):", X3_train.shape, "dtype:", X3_train.dtype)
print("X3_test shape (N, seq_len, channels):", X3_test.shape, "dtype:", X3_test.dtype)

print("y1:", None if y1 is None else (y1.shape, y1.dtype, "unique_classes:", np.unique(y1).tolist()))
print("y2:", None if y2 is None else (y2.shape, y2.dtype, "unique_classes:", np.unique(y2).tolist()))
print("y3_test:", None if y3_test is None else (y3_test.shape, y3_test.dtype, "unique_classes:", np.unique(y3_test).tolist()))
print("y3_train:", (y3_train.shape, y3_train.dtype, "unique_classes:", np.unique(y3_train).tolist()))

# sanity checks
for name, X in [("X1", X1), ("X2", X2), ("X3_train", X3_train)]:
    if X.ndim != 3:
        raise AssertionError(f"{name} is not 3D after prepare: ndim={X.ndim}")
    if X.shape[0] == 0:
        raise AssertionError(f"{name} has zero samples")

print("\nSanity checks passed.")
