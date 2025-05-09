import pandas as pd
import os

# NSL-KDD column headers (based on official documentation)
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level"
]

# Download URL for KDDTrain+ (use .txt as source)
url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"

# Read the dataset
print("[INFO] Downloading NSL-KDD dataset...")
df = pd.read_csv(url, names=columns)

# Drop difficulty level (not needed for training)
df.drop("difficulty_level", axis=1, inplace=True)

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save as CSV
df.to_csv("data/nsl_kdd.csv", index=False)
print("[INFO] Saved to data/nsl_kdd.csv ✅")
