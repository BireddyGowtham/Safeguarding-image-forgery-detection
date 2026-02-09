import os
from pathlib import Path
from io import BytesIO

from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import timm

# === Import your model class and transforms from st.py ===
from src.st import SwinTransformerForForgery, get_transforms

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model (use pathlib to avoid backslash escape issues)
MODEL_PATH = str(Path(__file__).resolve().parent / 'models' / 'best_model.pth')

model = SwinTransformerForForgery(
    model_name='swin_tiny_patch4_window7_224',
    num_classes=2,
    pretrained=False
).to(device)

checkpoint = None
if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        # support both {'model_state_dict': ...} and raw state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {MODEL_PATH}")
        elif isinstance(checkpoint, dict):
            try:
                model.load_state_dict(checkpoint)
                print(f"Loaded state_dict from {MODEL_PATH}")
            except Exception as e:
                print(f"Checkpoint at {MODEL_PATH} is not a model state dict: {e}")
        else:
            print(f"Checkpoint at {MODEL_PATH} has unexpected format (type={type(checkpoint)}). Skipping load.")
    except Exception as e:
        # torch.load failed (unpickling error, SSL/IO error, etc.)
        print(f"Failed to load checkpoint '{MODEL_PATH}': {type(e).__name__}: {e}")
        print("Proceeding with uninitialized model (pretrained=False).")
else:
    print(f"No checkpoint found at {MODEL_PATH}; proceeding with initialized model.")

model.eval()

# Use same validation transform as training code
_, val_transform = get_transforms()


def preprocess_image(pil_img):
    """Apply same preprocessing as validation."""
    img = val_transform(pil_img)
    img = img.unsqueeze(0)  # [1, 3, 224, 224]
    return img.to(device)


def predict_image(pil_img):
    """Run model on a PIL image, return label and confidence."""
    tensor = preprocess_image(pil_img)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item() * 100.0

    # Assuming label 0 = authentic, 1 = tampered (forged)
    if pred_class == 0:
        label = "Authentic"
    else:
        label = "Forged"

    return label, round(confidence, 2)


@app.route('/', methods=['GET'])
def index():
    # show upload step by default
    return render_template('index.html', active_step=0)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    # Save uploaded file for preview
    save_path = Path(app.config['UPLOAD_FOLDER']) / file.filename
    file.save(str(save_path))

    # Open with PIL for model
    pil_img = Image.open(save_path).convert('RGB')

    label, conf = predict_image(pil_img)

    return render_template(
        'index.html',
        result=label,
        confidence=conf,
        image_url=url_for('static', filename=f'uploads/{file.filename}'),
        active_step=1
    )


if __name__ == '__main__':
    # When debugging in VS Code, disable the reloader to avoid SystemExit(3)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)