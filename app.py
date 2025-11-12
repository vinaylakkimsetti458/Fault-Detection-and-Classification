"""
Flask Web UI for EfficientNet_B0-based PyTorch model (.pth) deployment.

Fixes from previous version:
- Now loads EfficientNet_B0 instead of ResNet18.
- Handles state_dict or torch.jit.load.
- Automatically adapts classifier for custom number of classes.
"""

import io
import os
import traceback
from PIL import Image
from flask import Flask, request, render_template_string, redirect, url_for, flash
from werkzeug.utils import secure_filename

import torch
import torchvision.transforms as T
import torchvision.models as models

# Config
MODEL_PATH = 'model_with_70%_Acc.pth' # Your uploaded model
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'replace-me-with-a-secure-key'

# Dummy labels — replace with your real dataset labels
LABELS = ['class_0 : Faulty', 'class_1 : Normal']
NUM_CLASSES = len(LABELS)

# Image preprocessing
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def try_load_efficientnet(model_path, num_classes=NUM_CLASSES):
    """Load EfficientNet_B0 from a TorchScript file or state_dict."""
    try:
        m = torch.jit.load(model_path, map_location=DEVICE)
        m.eval()
        return m
    except Exception:
        pass

    model = models.efficientnet_b0(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)

    try:
        state = torch.load(model_path, map_location=DEVICE)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f'Failed to load EfficientNet_B0 weights: {e}')


def load_model_at_startup():
    global MODEL
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}")
        return

    try:
        MODEL = try_load_efficientnet(MODEL_PATH)
        print('✅ EfficientNet_B0 model loaded successfully.')
    except Exception:
        print('❌ Failed to load model:')
        traceback.print_exc()
        MODEL = None


load_model_at_startup()


HTML = """
<!doctype html>
<title>EfficientNet_B0 Model Deploy</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
<div class="container mt-4">
  <h2>Deploy PyTorch EfficientNet_B0 Model</h2>
  {% if model_loaded %}
  <div class="alert alert-success">Model loaded successfully!</div>
  {% else %}
  <div class="alert alert-warning">Model not loaded. Ensure the .pth file is in correct path.</div>
  {% endif %}

  <form method=post enctype=multipart/form-data action="/predict">
    <label class="form-label">Upload image</label>
    <input class="form-control" type=file name=image required>
    <button class="btn btn-primary mt-3">Predict</button>
  </form>

  {% if result %}
  <hr>
  <h4>Prediction Result</h4>
  <div class="card p-3">
    <p><strong>Label:</strong> {{ result.label }}</p>
    <p><strong>Confidence:</strong> {{ result.confidence }}</p>
  </div>
  {% endif %}
</div>
"""


@app.route('/')
def index():
    return render_template_string(HTML, model_loaded=(MODEL is not None), result=None)


@app.route('/predict', methods=['POST'])
def predict():
    global MODEL
    if MODEL is None:
        flash('Model not loaded!')
        return redirect(url_for('index'))

    if 'image' not in request.files:
        flash('No image uploaded')
        return redirect(url_for('index'))

    file = request.files['image']
    if not allowed_file(file.filename):
        flash('Invalid file type')
        return redirect(url_for('index'))

    img = Image.open(file.stream).convert('RGB')
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(x)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_idx = torch.argmax(probs).item()
        label = LABELS[top_idx] if top_idx < len(LABELS) else f'class_{top_idx}'
        confidence = float(probs[top_idx])

    result = {"label": label, "confidence": round(confidence, 4)}
    return render_template_string(HTML, model_loaded=True, result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)