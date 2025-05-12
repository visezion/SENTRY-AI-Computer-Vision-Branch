# merge_all_datasets.py

import pandas as pd
import os
from src.utils import load_nsl_kdd, load_cicids2017, load_unsw_nb15
from src.config import BASE_DIR
import numpy as np

print("[INFO] Loading NSL-KDD...")
nsl_path = os.path.join(BASE_DIR, "data", "nsl_kdd.csv")
X_nsl, y_nsl = load_nsl_kdd()

print("[INFO] Loading CICIDS2017...")
cicids_path = os.path.join(BASE_DIR, "data", "cicids2017_combined.csv")
X_cic, y_cic = load_cicids2017(cicids_path)

print("[INFO] Loading UNSW-NB15...")
unsw_path = os.path.join(BASE_DIR, "data", "unsw_nb15.csv")
X_unsw, y_unsw = load_unsw_nb15(unsw_path)

print("[INFO] Aligning features by column intersection...")

# Convert arrays back to DataFrames for alignment
cols_nsl = [f"f{i}" for i in range(X_nsl.shape[1])]
cols_cic = [f"f{i}" for i in range(X_cic.shape[1])]
cols_unsw = [f"f{i}" for i in range(X_unsw.shape[1])]

# Assign synthetic column names
nsl_df = pd.DataFrame(X_nsl, columns=cols_nsl)
cic_df = pd.DataFrame(X_cic, columns=cols_cic)
unsw_df = pd.DataFrame(X_unsw, columns=cols_unsw)

# Find common features (simple way)
common_cols = list(set(nsl_df.columns) & set(cic_df.columns) & set(unsw_df.columns))

print(f"[INFO] Using {len(common_cols)} common features across all datasets.")

# Align
nsl_df = nsl_df[common_cols]
cic_df = cic_df[common_cols]
unsw_df = unsw_df[common_cols]

# Combine
X_all = pd.concat([nsl_df, cic_df, unsw_df], ignore_index=True)
y_all = np.concatenate([y_nsl, y_cic, y_unsw])

# Final output
combined_df = X_all.copy()
combined_df['label'] = y_all
out_path = os.path.join(BASE_DIR, "data", "sentry_combined.csv")
combined_df.to_csv(out_path, index=False)

print(f"✅ Unified dataset saved to {out_path}")
