
# 🔐 SENTRY-AI: Explainable AI-Powered Intrusion Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Stars](https://img.shields.io/github/stars/visezion/Sentry-AI?style=social)
![Forks](https://img.shields.io/github/forks/visezion/Sentry-AI?style=social)
![Contributors](https://img.shields.io/github/contributors/visezion/Sentry-AI)
[![DOI](https://zenodo.org/badge/DOI/10.1016/B978-0-443-26482-5.00009-2.svg)](https://doi.org/10.1016/B978-0-443-26482-5.00009-2)

> **SENTRY-AI** is an open-source, **explainable cybersecurity framework** that integrates  
> **deep learning, anomaly detection, and interpretability** to deliver **state-of-the-art intrusion detection**.  
> Built for **enterprises, researchers, and governments**, it balances **accuracy and trust** in AI-driven security.

---

## ✨ Highlights
- 🔀 **Hybrid AI Fusion** – CNN (GAF image encoding) + Variational Autoencoder (VAE).  
- 🔎 **Explainable AI (XAI)** – Grad-CAM for human-centric attack visualization.  
- 📊 **Cross-Dataset Benchmarks** – Validated on NSL-KDD, CICIDS2017, UNSW-NB15.  
- ⚡ **High Performance** – Achieved **99.99% accuracy** on UNSW-NB15.  
- 🖥️ **Real-Time Monitoring** – Flask + SocketIO dashboard prototype.  
- 🌍 **Open Source** – MIT licensed, reproducible, and extensible.  

---

## 📊 Performance Benchmarks

### Table 1 – Results (This Work)
| Dataset     | Model   | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------------|---------|----------|-----------|--------|----------|---------|
| **NSL-KDD** | CNN     | 90.13%   | 82.56%    | 99.99% | 90.44%   | 99.97%  |
|             | Fusion  | **99.00%** | **99.84%** | **98.01%** | **98.92%** | **99.81%** |
| **CICIDS2017** | CNN  | 90.35%   | 67.14%    | 99.99% | 80.33%   | 99.97%  |
|             | Fusion  | **95.55%** | **99.90%** | **77.48%** | **87.28%** | **99.95%** |
| **UNSW-NB15** | CNN  | 100.00%  | 100.00%   | 100.00% | 100.00%  | –       |
|             | Fusion  | **99.99%** | **100.00%** | **99.99%** | **100.00%** | – |

---

### Table 2 – Comparative Results with Prior Work
| Dataset     | Model                         | Accuracy | F1-Score | Reference                  |
|-------------|-------------------------------|----------|----------|----------------------------|
| NSL-KDD     | CNN-LSTM (Hybrid DL)          | 98.99    | 98.82    | Aljawarneh et al., 2018    |
|             | **SENTRY-AI (Fusion)**        | **99.00** | **98.92** | *This work*                |
| CICIDS2017  | CNN-MCL                       | 94.32    | –        | Lin et al., 2024           |
|             | Hybrid LSTM-AE                | 94.11    | 82.24    | Gupta et al., 2022         |
|             | **SENTRY-AI (Fusion)**        | **95.55** | **87.28** | *This work*                |
| UNSW-NB15   | GMM-WGAN-IDS                  | 87.70    | 85.44    | Alomari et al., 2022       |
|             | CNN-VAE (Semi-Supervised)     | 91.13    | 89.45    | Saeed et al., 2022         |
|             | **SENTRY-AI (Fusion)**        | **99.99** | **100.00** | *This work*                |

---

## 🔎 Explainability in Action

<p align="center">
  <img src="docs/images/gradcam_example.png" width="650"/>
</p>

> Using **Grad-CAM**, SENTRY-AI provides **heatmaps of anomaly sources**,  
> enabling **analysts to interpret why an alert was triggered**.  
> This bridges the gap between **black-box AI** and **human trust in cybersecurity**.

---

## 📂 Project Structure
```

Sentry-AI/
│── data/                # Dataset loaders & preprocessors
│── models/              # CNN, VAE, Fusion architectures
│── train.py             # Model training scripts
│── eval.py              # Evaluation scripts
│── gradcam.py           # Grad-CAM visualization
│── dashboard/           # Flask + SocketIO real-time dashboard
│── docs/                # Documentation, benchmarks, visuals
│── requirements.txt     # Dependencies
│── LICENSE              # MIT License
│── README.md            # This file

````

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/visezion/Sentry-AI.git
cd Sentry-AI
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train a model

```bash
python train.py --dataset NSL-KDD
```

### 4. Evaluate performance

```bash
python eval.py --dataset CICIDS2017
```

### 5. Launch dashboard

```bash
python dashboard/app.py
```

---

## 🏆 Recognition & Impact

* Published in **Elsevier (AIoT Book Chapter, 2024)**
  [DOI: 10.1016/B978-0-443-26482-5.00009-2](https://doi.org/10.1016/B978-0-443-26482-5.00009-2)
* Benchmarked against **state-of-the-art models**, outperforming prior works.
* Open-source adoption by **researchers, universities, and security engineers**.
* Part of evidence for **UK Global Talent Visa** under *Innovation Criterion*.

---

## 🌍 Community

* 👥 Contributors: Cybersecurity & AI researchers worldwide.
* 📢 Discussions: Use GitHub [Discussions](https://github.com/visezion/Sentry-AI/discussions) for Q\&A.
* 🛠️ Contributions welcome – see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📜 Citation

If you use this project in research, cite:

```bibtex
@incollection{oluwasusi2025sentry,
  title={Explainable AI-Powered Anomaly Detection: A Computer Vision Approach to Strengthening Human-Centric Cybersecurity},
  author={Victor Ayodeji Oluwasusi},
  booktitle={Artificial Intelligence of Things (AIoT)},
  publisher={Elsevier},
  year={2024},
  doi={10.1016/B978-0-443-26482-5.00009-2}
}
```

---

## 📬 Contact

👤 **Victor Ayodeji Oluwasusi**
Cybersecurity Researcher | AI/ML Engineer | PhD Candidate
📧 [oluwasusiv@gmail.com](mailto:oluwasusiv@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/victor-ayodeji-oluwasusi-059567157/) | [GitHub](https://github.com/visezion)

---

## ⚖️ License

This project is licensed under the **MIT License** – use, share, and build upon it freely.

```

---

## 🚀 Why this is **Stronger**
- **Professional look** → Badges (license, DOI, stars, forks, contributors).  
- **Research credibility** → Benchmark tables vs. prior works, citations, DOI link.  
- **Industry appeal** → Dashboard + reproducibility.  
- **Community building** → Discussions, contribution guide.  
- **GT Visa ready** → Highlights recognition, innovation, global adoption.  

---
