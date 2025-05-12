import time
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
    start_time = time.time()
    print("[DEBUG] Starting chunked CSV read...")

    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=100000, low_memory=False):
        chunks.append(chunk)
        print(f"[DEBUG] Loaded chunk with {len(chunk)} rows")

    df = pd.concat(chunks, ignore_index=True)
    print(f"[DEBUG] Finished reading CSV. Total rows: {len(df)}")

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

    end_time = time.time()
    print(f"[✅] Data loaded and preprocessed in {end_time - start_time:.2f} seconds.")

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



def transform_to_gaf(X, batch_size=1000):
    from pyts.image import GramianAngularField
    import time
    from tqdm import tqdm

    start = time.time()
    gaf = GramianAngularField(image_size=IMAGE_SIZE, method='summation')
    images = []

    print(f"[INFO] Transforming {len(X)} samples to GAF images in batches of {batch_size}...")

    for i in tqdm(range(0, len(X), batch_size), desc="GAF Batching"):
        batch = X[i:i + batch_size]
        batch_images = gaf.fit_transform(batch)
        images.extend(batch_images)

    print(f"[✅] GAF transformation complete in {time.time() - start:.2f} seconds.")
    return np.expand_dims(np.array(images, dtype=np.float32), axis=1)

def fuse_predictions(y_prob_vae, y_prob_cnn, method='avg', threshold=0.5):
    
    """
    Fuse predictions from VAE and CNN using the specified method.

    Parameters:
    - y_prob_vae: numpy array of anomaly scores/probabilities from the VAE
    - y_prob_cnn: numpy array of anomaly scores/probabilities from the CNN
    - method: 'avg' for average fusion, 'vote' for majority voting
    - threshold: threshold for binary classification

    Returns:
    - y_prob_fused: fused probability or score
    - y_pred_fused: final binary prediction
    """
    assert method in ['avg', 'vote'], "Fusion method must be 'avg' or 'vote'"
    assert len(y_prob_vae) == len(y_prob_cnn), "Probability arrays must be of same length"

    if method == 'avg':
        y_prob_fused = (y_prob_vae + y_prob_cnn) / 2
        y_pred_fused = (y_prob_fused > threshold).astype(int)
    elif method == 'vote':
        y_pred_vae = (y_prob_vae > threshold).astype(int)
        y_pred_cnn = (y_prob_cnn > threshold).astype(int)
        y_pred_fused = ((y_pred_vae + y_pred_cnn) >= 1).astype(int)
        y_prob_fused = (y_pred_vae + y_pred_cnn) / 2  # proxy for AUC

    return y_prob_fused, y_pred_fused
