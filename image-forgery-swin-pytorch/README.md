# 🔍 Image Forgery Detection using Swin Transformer

> A deep learning system for detecting image forgeries using the **Swin Transformer** architecture, trained on the CASIA datasets.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Inference](#model-inference)
- [Troubleshooting](#troubleshooting)
- [Project Workflow](#project-workflow)
- [Contributing](#contributing)
- [License](#license)

---

## 🧠 Overview

This project implements an image forgery detection system using the **Swin Transformer** architecture with **PyTorch**. The model is trained on the **CASIA v1 and v2 datasets**, which contain both authentic and tampered images, enabling binary classification of image integrity.

---

## 📁 Project Structure

```
image-forgery-swin-pytorch/
│
├── src/
│   ├── train.py          # Main entry point for training the model
│   ├── st.py             # Complete training script for image forgery detection
│   ├── dataset.py        # Dataset loading and preprocessing
│   ├── model.py          # Swin Transformer model definition
│   ├── evaluate.py       # Model evaluation functions
│   └── utils.py          # Utility functions for reproducibility and dataset downloading
│
├── configs/
│   └── default.yaml      # Configuration settings for hyperparameters and dataset paths
│
├── data/
│   ├── CASIA1/           # CASIA v1 dataset (authentic and tampered images)
│   └── CASIA2/           # CASIA v2 dataset (authentic and tampered images)
│
├── models/
│   └── best_model.pth    # Best model weights after training
│
├── notebooks/
│   └── exploration.ipynb # Jupyter notebook for exploratory data analysis
│
├── scripts/
│   └── run_train.sh      # Shell script to automate the training process
│
├── requirements.txt      # Required Python packages and their versions
├── .gitignore            # Files and directories to ignore by Git
└── README.md             # Project documentation
```

---

## 💻 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python    | 3.8+    | 3.10+       |
| RAM       | 4 GB    | 8 GB+       |
| VRAM      | 4 GB    | 8 GB+       |
| CUDA      | 11.0+   | 12.0+       |

> ⚠️ GPU acceleration is **optional** but strongly recommended for faster training.

---

## ⚙️ Installation

**1. Clone the repository:**

```bash
git clone <repository-url>
cd image-forgery-swin-pytorch
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Verify installation:**

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🗂️ Dataset Setup

**1. Download** the CASIA v1 and v2 datasets from the official sources.

**2. Extract** both datasets into the `data/` directory with the following structure:

```
data/
├── CASIA1/
│   ├── Au/               # Authentic images
│   └── Tp/               # Tampered images
│
└── CASIA2/
    ├── Au/               # Authentic images
    ├── Tp/               # Tampered images
    └── CASIA 2 Groundtruth/
```

---

## 🚀 Usage

### Option 1 — Quick Training

```bash
python train.py
```

This will:
- Load the CASIA datasets from the `data/` directory
- Train the Swin Transformer model
- Save the best model weights to `models/best_model.pth`
- Display training progress and metrics

---

### Option 2 — Shell Script (Linux / macOS)

```bash
bash scripts/run_train.sh
```

---

### Option 3 — Custom Training

```bash
python -c "from src.st import train; train()"
```

Or modify and run `src/st.py` directly to customize hyperparameters.

---

### Option 4 — Web Application

Once training is complete (or using the pre-trained `models/best_model.pth`):

```bash
python app.py
```

Then open your browser and navigate to:

```
http://localhost:5000
```

Upload an image through the web interface to get forgery detection results.

---

### Option 5 — Testing & Debugging

```bash
# Test on new images
python test_model.py

# Debug script
python debug_model.py
```

---

### Option 6 — Jupyter Notebook

For exploratory data analysis:

```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## 🔬 Model Inference

Run inference on a custom image using the following snippet:

```python
import torch
from src.st import SwinTransformerForForgery, get_transforms
from PIL import Image

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = SwinTransformerForForgery(
    model_name='swin_tiny_patch4_window7_224',
    num_classes=2,
    pretrained=False
)
model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
model.to(device).eval()

# Load and preprocess image
transforms = get_transforms()
image = Image.open('path/to/image.jpg')
image_tensor = transforms(image).unsqueeze(0).to(device)

# Run prediction
with torch.no_grad():
    output = model(image_tensor)
    prediction = torch.argmax(output, dim=1).item()
    confidence = torch.softmax(output, dim=1).max().item()

label = 'Tampered' if prediction == 1 else 'Authentic'
print(f"Prediction : {label}")
print(f"Confidence : {confidence:.4f}")
```

---

## 🛠️ Configuration

Edit `configs/default.yaml` to customize training behavior:

| Parameter          | Description                         |
|--------------------|-------------------------------------|
| `model_name`       | Swin Transformer variant            |
| `learning_rate`    | Optimizer learning rate             |
| `batch_size`       | Number of samples per batch         |
| `num_epochs`       | Total training epochs               |
| `dataset_path`     | Path to CASIA dataset               |
| `device`           | `cpu` or `cuda`                     |

---

## 🧯 Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `batch_size` in config or switch to CPU |
| `Dataset not found` | Ensure CASIA datasets are extracted to `data/` |
| `Module import errors` | Re-run `pip install -r requirements.txt` |
| `Flask port in use` | Change port in `app.py` or kill the process on port 5000 |

---

## 🔄 Project Workflow

```
1. Prepare   →  Download and organize CASIA datasets
2. Install   →  Set up Python environment with requirements
3. Train     →  Run training script to train the model
4. Evaluate  →  Test model performance on the validation set
5. Deploy    →  Use the web app for real-time predictions
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- 🐛 Open an issue for bugs or feature requests
- 🔧 Submit a pull request with improvements
- 📖 Improve documentation

Please follow standard GitHub contribution guidelines.

---

## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for full details.

---

<div align="center">
  <sub>Built with ❤️ using PyTorch and Swin Transformer</sub>
</div>
