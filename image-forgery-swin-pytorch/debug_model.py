import os
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import timm
from torchvision import transforms
from src.st import SwinTransformerForForgery, get_transforms

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
MODEL_PATH = str(Path(__file__).resolve().parent / 'models' / 'best_model.pth')
print(f"\nLoading model from: {MODEL_PATH}")
print(f"Model exists: {os.path.exists(MODEL_PATH)}")

model = SwinTransformerForForgery(
    model_name='swin_tiny_patch4_window7_224',
    num_classes=2,
    pretrained=False
).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
print(f"\nCheckpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Raw state dict'}")
print(f"Checkpoint type: {type(checkpoint)}")

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Loaded model_state_dict")
    if 'metrics' in checkpoint:
        print(f"Training metrics: {checkpoint['metrics']}")
elif isinstance(checkpoint, dict):
    model.load_state_dict(checkpoint)
    print("✓ Loaded raw state dict")

model.eval()
print("✓ Model set to eval mode")

# Test with dummy image
print("\n" + "="*60)
print("Testing with dummy images")
print("="*60)

_, val_transform = get_transforms()

# Create a dummy authentic-looking image (smooth, uniform)
dummy_authentic = Image.new('RGB', (224, 224), color=(128, 128, 128))
dummy_authentic.save('test_authentic.jpg')

# Create a forged-looking image (random noise/mixed patterns)
import numpy as np
dummy_forged_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
dummy_forged = Image.fromarray(dummy_forged_array)
dummy_forged.save('test_forged.jpg')

for test_name, test_image in [('Authentic (smooth)', dummy_authentic), ('Forged (noisy)', dummy_forged)]:
    tensor = val_transform(test_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item() * 100.0
        
        print(f"\n{test_name}")
        print(f"  Raw outputs: {outputs[0].cpu().numpy()}")
        print(f"  Probabilities: [class_0={probs[0]:.4f}, class_1={probs[1]:.4f}]")
        print(f"  Predicted class: {pred_class}")
        print(f"  Confidence: {confidence:.2f}%")
        
        # Try both interpretations
        if pred_class == 0:
            label_v1 = "Authentic"
            label_v2 = "Forged"
        else:
            label_v1 = "Forged"
            label_v2 = "Authentic"
        
        print(f"  If class_0=Authentic, class_1=Forged: {label_v1}")
        print(f"  If class_0=Forged, class_1=Authentic: {label_v2}")

print("\n" + "="*60)
print("Check your training code to verify class mapping:")
print("Look for: train_dataset._load_samples() or similar")
print("Which class (0 or 1) represents forged/tampered images?")
print("="*60)
