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

## ⚙️ Features

### ✅ Detection Capabilities
- CNN trained on GAF-transformed tabular data
- VAE for anomaly scoring
- CNN + VAE fusion evaluation
- Grad-CAM explainability
- Works with NSL-KDD, CICIDS2017, UNSW-NB15 datasets

### 📊 Real-Time Dashboard
- System status and alert panel
- Metrics summary (Accuracy, Precision, Recall, F1, AUC-ROC)
- Dataset and model metadata
- Grad-CAM visualizations with heatmaps
- Top detected attack types with bar/pie chart
- Resource monitoring (CPU, RAM, Disk)
- Full/mini-batch evaluation toggle
- Downloadable evaluation reports

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/visezion/SENTRY-AI.git
cd SENTRY-AI
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Place Your Dataset

Download and place datasets in the `data/` directory:

* `nsl_kdd_combined.csv`
* `cicids2017_combined.csv`
* `unsw_nb15_combined.csv`

### 4. Run Training/Evaluation

```bash
python -m src.main --dataset NSL-KDD
python -m src.main --dataset CICIDS2017
python -m src.main --dataset UNSW-NB15
```

---

## 🖥️ Launch Dashboard (comming soon)

```bash
python app.py
```

Then open your browser at [http://localhost:5000](http://localhost:5000)

---

## 🔧 Configuration

Edit `src/config.py` to control:

* Model parameters (epochs, batch size, learning rate)
* Image size for GAF transform
* Evaluation mode:

```python
USE_MINI_EVAL = True  # Set to False to use full evaluation
```

---

## 📄 Outputs

* Evaluation metrics: `outputs/metrics_<dataset>.json`
* Grad-CAM images: `outputs/gradcam_heatmaps/<dataset>/`
* Fusion reports: `outputs/fusion_metrics_<dataset>.txt`

---

## 🧪 Datasets Used

| Dataset    | Type          | Source               |
| ---------- | ------------- | -------------------- |
| NSL-KDD    | Tabular       | KDD Cup (Improved)   |
| CICIDS2017 | Network Flows | Canadian Institute   |
| UNSW-NB15  | Realistic Mix | Australian Cyber Lab |

---

## 📜 License

MIT License © 2025 \[Victor Ayodeji Oluwasusi]

---

## 🤝 Contribute

Pull requests are welcome. For major changes, please open an issue first.

---

## 📬 Contact

For questions or support, contact:

Victor Ayodeji Oluwasusi
PhD Researcher, Cybersecurity & AI
[GitHub](https://github.com/visezion) | [Scholar](https://scholar.google.com/citations?user=eeexwhIAAAAJ)
