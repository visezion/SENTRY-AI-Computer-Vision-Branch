import os
import pandas as pd
import glob

def merge_cicids(input_dir="data/cicids2017", output_path="data/cicids2017_combined.csv"):
    csvs = glob.glob(os.path.join(input_dir, "*.csv"))
    dfs = [pd.read_csv(f, low_memory=False) for f in csvs]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"✅ Merged CICIDS2017 into {output_path}")

if __name__ == "__main__":
    merge_cicids()
