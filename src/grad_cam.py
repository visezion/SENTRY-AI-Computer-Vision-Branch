# grad_cam.py (with full normalization + GAF transformation)

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split

from .config import MODEL_SAVE_PATH, GRADCAM_OUTPUT_DIR, DATASET_NAME, IMAGE_SIZE
from .utils import load_dataset, normalize_and_save, transform_to_gaf
from .models import AdvancedCNN


def apply_grad_cam(index=0):
    os.makedirs(GRADCAM_OUTPUT_DIR, exist_ok=True)

    print(f"[INFO] Applying Grad-CAM for dataset: {DATASET_NAME} at index {index}")

    # Load model
    model_name = f"cnn_model_{DATASET_NAME.lower().replace('-', '_')}.pth"
    model_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), model_name)

    model = AdvancedCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Load + normalize + GAF transform
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = normalize_and_save(X_train, X_test)
    X_test_gaf = transform_to_gaf(X_test)

    # Target image and label
    image = torch.tensor(X_test_gaf[index:index+1], requires_grad=True)
    label = y_test[index]

    # Hook to capture feature maps
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    handle = model.features[-1].register_forward_hook(forward_hook)

    # Forward pass
    output = model(image)
    pred_class = output.argmax(dim=1).item()

    # Backprop to get gradients
    model.zero_grad()
    class_score = output[0, pred_class]
    class_score.backward()

    # Get gradients and feature maps
    grads = image.grad.data[0].numpy()        # shape: (1, 28, 28)
    fmap = activations[0].detach().numpy()[0] # shape: (num_channels, H, W)

    # Global average pooling on gradients
    weights = np.mean(grads, axis=(1, 2)) if grads.ndim == 3 else np.mean(grads)

    # Weighted combination of feature maps
    heatmap = np.zeros_like(fmap[0])
    for i, w in enumerate(weights):
        heatmap += w * fmap[i]

    # Apply ReLU and normalize
    heatmap = np.maximum(heatmap, 0)
    heatmap = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)


    # Overlay on GAF
    gaf = image[0][0].detach().numpy()
    overlay = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    gaf_rgb = cv2.cvtColor(np.uint8(255 * gaf), cv2.COLOR_GRAY2BGR)
    superimposed = cv2.addWeighted(gaf_rgb, 0.6, overlay, 0.4, 0)

    # Save visualization
    output_path = os.path.join(GRADCAM_OUTPUT_DIR, f"{DATASET_NAME.lower()}_gradcam_{index}.png")
    plt.imshow(superimposed)
    plt.title(f"Grad-CAM: Pred={pred_class} | True={label}")
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()

    print(f"[✅] Grad-CAM saved to {output_path}")


if __name__ == '__main__':
    apply_grad_cam(index=0)
