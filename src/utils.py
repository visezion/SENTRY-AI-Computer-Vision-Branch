import pandas as pd
import numpy as np
import os
import glob
from .config import DATASET_NAME, DATA_PATH, SCALER_PATH
from sklearn.preprocessing import MinMaxScaler
import joblib
from pyts.image import GramianAngularField
from .config import IMAGE_SIZE



def normalize_and_save(X_train, X_val):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save the fitted scaler
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    
    return X_train_scaled, X_val_scaled


def load_nsl_kdd(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    df.columns = df.columns.str.strip()
    df['label'] = df['label'].apply(lambda x: 0 if str(x).lower() == 'normal' else 1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.select_dtypes(include=[np.number])
    X = df.drop(columns=['label'], errors='ignore')
    y = df['label'].values
    return X.values, y


def load_cicids2017(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    df.dropna(inplace=True)
    df.columns = df.columns.str.strip()
    df['Label'] = df['Label'].astype(str)
    df['label'] = df['Label'].apply(lambda x: 0 if 'benign' in x.lower() else 1)
    df.drop(columns=['Label'], inplace=True)

    drop_cols = [
        'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port',
        'Timestamp', 'Protocol', 'SimillarHTTP', 'Flow Bytes/s', 'Flow Packets/s'
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore', inplace=True)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.select_dtypes(include=[np.number])
    X = df.drop(columns=['label'], errors='ignore')
    y = df['label'].values
    return X.values, y


def load_unsw_nb15(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    df.columns = df.columns.str.strip()
    
    if 'label' in df.columns:
        df['label'] = df['label'].apply(lambda x: 0 if str(x).lower() == 'normal' else 1)
    elif 'attack_cat' in df.columns:
        df['label'] = df['attack_cat'].apply(lambda x: 0 if pd.isna(x) or x == 'Normal' else 1)
        df.drop(columns=['attack_cat'], inplace=True)

    drop_cols = ['id', 'proto', 'state', 'service', 'source_ip', 'destination_ip']
    df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore', inplace=True)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.select_dtypes(include=[np.number])
    X = df.drop(columns=['label'], errors='ignore')
    y = df['label'].values
    return X.values, y


def merge_unsw_parts(input_dir="data/unsw", output_path="data/unsw_nb15.csv"):
    csvs = glob.glob(os.path.join(input_dir, "UNSW-NB15_*.csv"))
    dfs = [pd.read_csv(f) for f in csvs]
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(output_path, index=False)
    print(f"✅ UNSW-NB15 merged into {output_path}")

def load_dataset():
    dataset = DATASET_NAME.strip().lower()

    if "nsl" in dataset:
        print("[INFO] Routing to load_nsl_kdd()")
        return load_nsl_kdd(DATA_PATH)
    elif "unsw" in dataset:
        print("[INFO] Routing to load_unsw_nb15()")
        return load_unsw_nb15(DATA_PATH)
    elif "cic" in dataset or "cicids" in dataset:
        print("[INFO] Routing to load_cicids2017()")
        return load_cicids2017(DATA_PATH)
    elif "sentry" in dataset or "combined" in dataset:
        print("[INFO] Routing to load_cicids2017() (combined)")
        return load_cicids2017(DATA_PATH)
    else:
        raise ValueError(f"❌ Unknown dataset: {DATASET_NAME}")





def transform_to_gaf(X):
    gaf = GramianAngularField(image_size=IMAGE_SIZE, method='difference')
    return np.array([gaf.fit_transform(x.reshape(1, -1))[0] for x in X], dtype=np.float32)[:, np.newaxis, :, :]
