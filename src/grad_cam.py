import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from .config import MODEL_SAVE_PATH, GRADCAM_OUTPUT_DIR
from .utils import load_nsl_kdd, normalize_and_save, transform_to_gaf, IMAGE_SIZE
from .train import CNNModel

def apply_grad_cam(index=0):
    os.makedirs(GRADCAM_OUTPUT_DIR, exist_ok=True)
    model = CNNModel()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu')))
    model.eval()

    X, y = load_nsl_kdd()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = normalize_and_save(X_train, X_test)
    X_test_gaf = transform_to_gaf(X_test)

    image = torch.tensor(X_test_gaf[index:index+1], requires_grad=True)
    label = y_test[index]

    model.eval()
    feature_maps = []
    def forward_hook(module, input, output):
        feature_maps.append(output)

    handle = model.features[-2].register_forward_hook(forward_hook)
    output = model(image)
    pred_class = output.argmax(dim=1).item()

    model.zero_grad()
    class_score = output[0, pred_class]
    class_score.backward()

    gradients = image.grad[0][0].numpy()
    fmap = feature_maps[0][0].detach().numpy().mean(axis=0)
    heatmap = np.maximum(fmap * gradients, 0)
    heatmap = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    gaf = image[0][0].detach().numpy()
    overlay = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    gaf_rgb = cv2.cvtColor(np.uint8(255 * gaf), cv2.COLOR_GRAY2BGR)
    superimposed = cv2.addWeighted(gaf_rgb, 0.6, overlay, 0.4, 0)

    plt.imshow(superimposed)
    plt.title(f"Grad-CAM for Prediction: {pred_class} | True: {label}")
    output_path = os.path.join(GRADCAM_OUTPUT_DIR, f"gradcam_{index}.png")
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Grad-CAM saved to {output_path}")

if __name__ == '__main__':
    apply_grad_cam(index=0)