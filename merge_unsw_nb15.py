import pandas as pd
import glob
import os

# Folder where you put the 4 CSVs
unsw_folder = os.path.join("data", "unsw")

# Match all relevant UNSW CSV files
csv_files = glob.glob(os.path.join(unsw_folder, "UNSW-NB15_*.csv"))

# Read and concatenate
df_list = [pd.read_csv(f) for f in csv_files]
combined = pd.concat(df_list, ignore_index=True)

# Save merged file
combined_path = os.path.join("data", "unsw_nb15.csv")
combined.to_csv(combined_path, index=False)

print(f"✅ Combined UNSW-NB15 dataset saved to: {combined_path}")
