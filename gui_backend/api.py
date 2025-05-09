# gui_backend/api.py

from flask import Flask, request, jsonify, send_file
import os
from src.config import MODEL_SAVE_PATH, GRADCAM_OUTPUT_DIR
from src.evaluate import evaluate_model
from src.fusion import evaluate_fusion
from src.grad_cam import apply_grad_cam
from src.utils import load_nsl_kdd, normalize_and_save, transform_to_gaf, IMAGE_SIZE
from src.train import AdvancedCNN
import torch
import torch.nn.functional as F
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

@app.route("/metrics", methods=["GET"])
def metrics():
    import sys
    from io import StringIO
    temp_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    evaluate_model()
    sys.stdout = temp_stdout
    return jsonify({"metrics": captured.getvalue()})

@app.route("/fusion", methods=["GET"])
def fusion():
    import sys
    from io import StringIO
    temp_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    evaluate_fusion()
    sys.stdout = temp_stdout
    return jsonify({"fusion_metrics": captured.getvalue()})

@app.route("/explain/<int:index>", methods=["GET"])
def explain(index):
    apply_grad_cam(index)
    img_path = os.path.join(GRADCAM_OUTPUT_DIR, f"gradcam_{index}.png")
    return send_file(img_path, mimetype='image/png')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array(data["features"], dtype=np.float32).reshape(1, -1)
    scaler = torch.load(os.path.join("models", "scaler.pkl"), map_location=torch.device("cpu"))
    features = scaler.transform(features)
    gaf_image = transform_to_gaf(features)[0:1]  # shape: (1, 1, H, W)

    model = AdvancedCNN()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device("cpu")))
    model.eval()

    with torch.no_grad():
        outputs = model(torch.tensor(gaf_image))
        probs = F.softmax(outputs, dim=1).numpy()
        pred_class = int(np.argmax(probs))
        confidence = float(np.max(probs))

    return jsonify({"prediction": pred_class, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True, port=8000)
