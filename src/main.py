from src.train import train_model
from src.evaluate import evaluate_model
from src.grad_cam import apply_grad_cam
from src.fusion import evaluate_fusion

if __name__ == '__main__':
    print("[1] Training CNN Model...")
    train_model()

    print("\n[2] Evaluating CNN Performance...")
    evaluate_model()

    print("\n[3] Generating Grad-CAM Visualization (index=0)...")
    apply_grad_cam(index=0)

    print("\n[4] Running CNN + VAE Fusion Evaluation...")
    evaluate_fusion()

    print("\n✅ SENTRY-AI Evaluation Complete")
