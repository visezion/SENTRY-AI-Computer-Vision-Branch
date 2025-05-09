# 🧠 SENTRY-AI: Explainable AI-Powered Anomaly Detection

SENTRY-AI is a modular, computer vision–integrated Intrusion Detection System (IDS) built with:

* CNN-based anomaly detection (GAF image encoding)
* Variational Autoencoder (VAE) for unsupervised analysis
* Grad-CAM explainability for human-centric validation
* Multi-dataset training: NSL-KDD, CICIDS2017, UNSW-NB15
* Fusion-based decision analysis (CNN + VAE)

---

## 📁 Project Structure

```
SENTRY-AI--Computer-Vision-Branch/
│
├── data/                         # Raw + processed datasets
│   ├── nsl_kdd.csv
│   ├── cicids2017_combined.csv
│   ├── unsw_nb15.csv
│   └── sentry_combined.csv
│
├── models/                      # Trained model weights
│   ├── cnn_model_<dataset>.pth
│   ├── vae_model_<dataset>.pth
│
├── outputs/                     # Metrics + Grad-CAMs
│   ├── fusion_metrics_<dataset>.txt
│   └── gradcam_heatmaps/<dataset>/
│
├── src/
│   ├── main.py                  # Entry point
│   ├── train.py                 # CNN model training
│   ├── vae_train.py             # VAE model training
│   ├── fusion.py                # Fusion evaluation
│   ├── grad_cam.py              # Grad-CAM visualization
│   ├── evaluate.py              # CNN metrics evaluation
│   ├── config.py                # Central config + dataset toggle
│   ├── utils.py                 # Loaders, preprocessors
│   └── models.py                # CNN, VAE architectures
│
└── gui_frontend/                # Optional Vite/Tailwind React GUI
```

---

## 🚀 Run Full Pipeline

```bash
python -m src.main --dataset NSL-KDD
python -m src.main --dataset CICIDS2017
python -m src.main --dataset UNSW-NB15
python -m src.main --dataset SENTRY-COMBINED
```

Each run will:

* 🔁 Train (or load) CNN & VAE models per dataset
* 📊 Evaluate performance (accuracy, F1, etc.)
* 🔍 Generate Grad-CAM explanation
* 🤖 Run CNN + VAE fusion
* 💾 Save all results to `outputs/`

---

## 📥 Dataset Preparation

### NSL-KDD

Place in:

```
data/nsl_kdd.csv
```

### CICIDS2017

1. Download `MachineLearningCSV.zip`
2. Merge all CSVs into:

```
data/cicids2017_combined.csv
```

### UNSW-NB15

1. Download `UNSW-NB15_1.csv` to `UNSW-NB15_4.csv`
2. Merge using:

```python
from src.utils import merge_unsw_parts
merge_unsw_parts()
```

### Combined (All)

Run:

```bash
python merge_all_datasets.py
```

---

## 🧠 Model Checkpointing

* All models are auto-saved and reused per dataset.
* You’ll be prompted to reuse saved models or re-train:

```
models/cnn_model_nsl_kdd.pth
models/vae_model_unsw_nb15.pth
```

---

## 🖼️ Grad-CAM Visualization

* Output in:

```
outputs/gradcam_heatmaps/<dataset>/gradcam_0.png
```

Use `--gradcam_index N` to select which input sample to visualize.

---

## 📊 Metric Logs

All evaluation scores (CNN, VAE, Fusion) are saved to:

```
outputs/fusion_metrics_<dataset>.txt
```

---

## 📡 Future Additions

* 🧩 Live network capture
* 🌐 GUI switching between datasets + live feedback
* 📦 Docker/streamlit deployment

---

## 🧑‍💻 Author

Victor Ayodeji Oluwasusi
PhD Researcher, Cybersecurity & AI
[GitHub](https://github.com/visezion) | [Scholar](https://scholar.google.com/citations?user=eeexwhIAAAAJ)

---

## 📜 License

MIT — open source and reproducible for academic and commercial use.
