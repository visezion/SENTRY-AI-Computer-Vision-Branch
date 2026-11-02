
# 🔐 SENTRY-AI: Explainable AI-Powered Intrusion Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Stars](https://img.shields.io/github/stars/visezion/Sentry-AI?style=social)
![Forks](https://img.shields.io/github/forks/visezion/Sentry-AI?style=social)

> **SENTRY-AI** is an open-source, **explainable cybersecurity system** that fuses  
> **Convolutional Neural Networks (CNNs)** and **Variational Autoencoders (VAEs)**  
> with **Grad-CAM explainability**, achieving **state-of-the-art intrusion detection**  
> on benchmark datasets (NSL-KDD, CICIDS2017, UNSW-NB15).

---

## 🚀 Highlights
- **Hybrid AI Fusion** – CNN (Gramian Angular Fields) + VAE anomaly scoring.  
- **Explainability** – Grad-CAM heatmaps for human-centric attack analysis.  
- **Cross-Dataset Benchmarks** – Outperforms published state-of-the-art methods.  
- **High Performance** – Achieved **99.99% accuracy** on UNSW-NB15.  
- **Open Source** – MIT licensed for research and enterprise use.  

---

## 📊 Benchmarks

### Results (This Work)
| Dataset     | Model   | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------------|---------|----------|-----------|--------|----------|---------|
| **UNSW-NB15** | CNN  | 100.00%  | 100.00%   | 100.00% | 100.00%  | –       |
|             | Fusion  | **99.99%** | **100.00%** | **99.99%** | **100.00%** | – |

### Comparison with Prior Work
| Dataset     | Model                         | Accuracy | F1-Score | Reference                  |
|-------------|-------------------------------|----------|----------|----------------------------|
| UNSW-NB15   | GMM-WGAN-IDS                  | 87.70    | 85.44    | Alomari et al., 2022       |
|             | CNN-VAE (Semi-Supervised)     | 91.13    | 89.45    | Saeed et al., 2022         |
|             | **SENTRY-AI (Fusion)**        | **99.99** | **100.00** | *This work*                |

Results on the UNSW-NB15 benchmark; real-world performance may vary.” Experts know that dataset often produces near-perfect scores, and showing you know it too adds credibility.
---

## 🔎 Explainability

<p align="center">
  <img src="docs/images/gradcam_example.png" width="650"/>
</p>

> **Grad-CAM heatmaps** highlight suspicious traffic patterns,  
> making intrusion detection **transparent and interpretable**.

---

## 📂 Project Structure
```

Sentry-AI/
│── models/              # CNN, VAE, Fusion architectures
│── train.py             # Training script
│── eval.py              # Evaluation script
│── gradcam.py           # Explainability module
│── dashboard/           # Real-time Flask dashboard
│── docs/                # Benchmarks & visuals
│── requirements.txt     # Dependencies
│── LICENSE              # MIT License
│── README.md            # Project overview

````


---

## 📜 Citation
```bibtex
@incollection{oluwasusi2025sentry,
  title={Explainable AI-Powered Anomaly Detection: A Computer Vision Approach to Strengthening Human-Centric Cybersecurity},
  author={Victor Ayodeji Oluwasusi},
}
````

---

## 📬 Contact

👤 **Victor Ayodeji Oluwasusi**
Cybersecurity Researcher | AI/ML Engineer | PhD Candidate
📧 [oluwasusiv@gmail.com](mailto:oluwasusiv@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/victor-ayodeji-oluwasusi-059567157/) | [GitHub](https://github.com/visezion)

---

## ⚖️ License

MIT License – free for research, academic, and enterprise use.

```
