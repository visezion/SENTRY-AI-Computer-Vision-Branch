import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pyts.image import GramianAngularField
import joblib
import os
from .config import DATA_PATH, SCALER_PATH, IMAGE_SIZE

def load_nsl_kdd():
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)
    # Encode labels: normal = 0, attack = 1
    df['label'] = df['label'].apply(lambda x: 0 if x.lower() == 'normal' else 1)
    
    # Drop symbolic features and label
    X = df.drop(columns=['label', 'protocol_type', 'service', 'flag'], errors='ignore')
    y = df['label'].values
    return X.values, y

def normalize_and_save(X_train, X_val):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    joblib.dump(scaler, SCALER_PATH)
    return X_train, X_val

def transform_to_gaf(X):
    gaf = GramianAngularField(image_size=IMAGE_SIZE, method='difference')
    return np.array([gaf.fit_transform(x.reshape(1, -1))[0] for x in X], dtype=np.float32)[:, np.newaxis, :, :]
