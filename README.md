# 🧠 SENTRY-AI: Explainable AI for Intrusion Detection

**SENTRY-AI** is an **AI-powered Intrusion Detection System (IDS)** designed to advance cybersecurity research and practice.
Built with **deep learning, explainable AI, and fusion models**, it achieved **99.99% accuracy on the UNSW-NB15 benchmark dataset** - a state-of-the-art performance for anomaly detection.

This project was created **independently and voluntarily** to support the global cybersecurity community. It is licensed under **MIT** for open academic and industry adoption.

---

## ✨ Why SENTRY-AI Matters

Cybersecurity threats are growing in scale and complexity. Traditional IDS models struggle with accuracy and transparency.

**SENTRY-AI introduces:**

* **High performance** (99.99% on UNSW-NB15, surpassing baseline methods).
* **Explainability** (Grad-CAM anomaly visualisation for human validation).
* **Multi-dataset robustness** (tested on NSL-KDD, CICIDS2017, UNSW-NB15).
* **Open-source availability** for researchers, enterprises, and startups.

📊 *This work contributes to advancing the digital technology sector by providing a reproducible, high-accuracy IDS framework available to the global community.*

---

## ⚡ Key Features

* **CNN anomaly detection** with Gramian Angular Field (GAF) encoding.
* **Variational Autoencoder (VAE)** for unsupervised anomaly detection.
* **Fusion-based decision engine** combining CNN + VAE.
* **Grad-CAM explainability** → transparent AI, trusted by human operators.
* **Dashboard (coming soon)** → metrics, anomaly heatmaps, system monitoring.

---

## 📊 Benchmark Performance

| Dataset    | Accuracy   | Precision | Recall | F1-Score |
| ---------- | ---------- | --------- | ------ | -------- |
| UNSW-NB15  | **99.99%** | 99.98%    | 99.99% | 99.98%   |
| CICIDS2017 | 98.7%      | 98.4%     | 98.6%  | 98.5%    |
| NSL-KDD    | 97.9%      | 97.5%     | 97.8%  | 97.6%    |

---

## 🚀 Quick Start

```bash
# Clone repo
git clone https://github.com/visezion/SENTRY-AI.git
cd SENTRY-AI

# Install dependencies
pip install -r requirements.txt

# Train/evaluate
python -m src.main --dataset UNSW-NB15
```

---

## 📁 Repository Overview

* **data/** – benchmark datasets (NSL-KDD, CICIDS2017, UNSW-NB15)
* **src/** – training, evaluation, explainability (CNN, VAE, Fusion)
* **models/** – saved model weights
* **outputs/** – metrics & Grad-CAM heatmaps
* **gui\_frontend/** – optional dashboard (React + Vite + Tailwind)

---

## 🌍 Sector Impact & Adoption

* ⭐ 6 stars, 🍴 3 forks → independent recognition by researchers/developers.
* Used as a **reference model for benchmarking IDS research**.
* Freely available under MIT license for academic labs and enterprises.
* Relevant to **UK cybersecurity priorities** (resilience, AI adoption, network security).

---

## 📜 License

MIT License © 2025 - **Victor Ayodeji Oluwasusi**

---

## 📬 Contact

**Victor Ayodeji Oluwasusi**
PhD Researcher, Cybersecurity & AI

🔗 [GitHub](https://github.com/visezion) | [Google Scholar](https://scholar.google.com/citations?user=eeexwhIAAAAJ)
