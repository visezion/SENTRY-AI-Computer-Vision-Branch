# src/main.py

import argparse
from src import config
from src.train import train_model
from src.evaluate import evaluate_model
from src.grad_cam import apply_grad_cam
from src.fusion import evaluate_fusion

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SENTRY-AI on specified dataset")
    parser.add_argument("--dataset", type=str, default="NSL-KDD", help="Dataset to use: NSL-KDD, CICIDS2017, UNSW-NB15, SENTRY-COMBINED")
    parser.add_argument("--gradcam_index", type=int, default=0, help="Index for Grad-CAM visualization")
    args = parser.parse_args()

    config.DATASET_NAME = args.dataset
    print(f"\n📊 Running SENTRY-AI for dataset: {config.DATASET_NAME}\n")

    print("[1] Training CNN Model...")
    train_model()

    print("\n[2] Evaluating CNN Performance...")
    evaluate_model()

    print(f"\n[3] Generating Grad-CAM Visualization (index={args.gradcam_index})...")
    apply_grad_cam(index=args.gradcam_index)

    print("\n[4] Running CNN + VAE Fusion Evaluation...")
    evaluate_fusion()

    print("\n✅ SENTRY-AI Evaluation Complete\n")
