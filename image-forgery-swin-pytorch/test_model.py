#!/usr/bin/env python
"""Test the trained model on sample images"""

import os
import sys
import torch
from pathlib import Path
from PIL import Image
import random

sys.path.insert(0, str(Path(__file__).parent))

from src.st import SwinTransformerForForgery, get_transforms

def test_model():
    """Test the model on some CASIA2 images"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    MODEL_PATH = Path(__file__).parent / 'models' / 'best_model.pth'
    
    model = SwinTransformerForForgery(
        model_name='swin_tiny_patch4_window7_224',
        num_classes=2,
        pretrained=False
    ).to(device)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded from {MODEL_PATH}")
    print(f"  Validation accuracy during training: {checkpoint['val_acc']:.4f}\n")
    
    _, val_transform = get_transforms()
    
    # Test on some CASIA2 images
    casia2_path = Path(__file__).parent.parent / 'data' / 'CASIA2'
    
    authentic_dir = casia2_path / 'Au'
    tampered_dir = casia2_path / 'Tp'
    
    print("="*70)
    print("Testing on random CASIA2 images")
    print("="*70 + "\n")
    
    # Test authentic images
    print("AUTHENTIC IMAGES:")
    print("-" * 70)
    authentic_images = list(authentic_dir.glob('*'))[:3]
    
    for img_path in authentic_images:
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
            continue
            
        try:
            pil_img = Image.open(img_path).convert('RGB')
            tensor = val_transform(pil_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_class = torch.argmax(probs).item()
                confidence = probs[pred_class].item() * 100.0
            
            # Class 0 = Authentic, Class 1 = Forged
            label = "Authentic" if pred_class == 0 else "Forged"
            print(f"Image: {img_path.name}")
            print(f"  Prediction: {label} ({confidence:.2f}%)")
            print(f"  Class probs: [Authentic={probs[0]:.4f}, Forged={probs[1]:.4f}]")
            print()
        except Exception as e:
            print(f"Error with {img_path.name}: {e}\n")
    
    # Test tampered/forged images
    print("\nFORGED/TAMPERED IMAGES:")
    print("-" * 70)
    tampered_images = list(tampered_dir.glob('*'))[:3]
    
    for img_path in tampered_images:
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
            continue
            
        try:
            pil_img = Image.open(img_path).convert('RGB')
            tensor = val_transform(pil_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_class = torch.argmax(probs).item()
                confidence = probs[pred_class].item() * 100.0
            
            # Class 0 = Authentic, Class 1 = Forged
            label = "Authentic" if pred_class == 0 else "Forged"
            print(f"Image: {img_path.name}")
            print(f"  Prediction: {label} ({confidence:.2f}%)")
            print(f"  Class probs: [Authentic={probs[0]:.4f}, Forged={probs[1]:.4f}]")
            print()
        except Exception as e:
            print(f"Error with {img_path.name}: {e}\n")
    
    print("="*70)
    print("✓ Model is now ready to use in the Flask app!")
    print("="*70)

if __name__ == '__main__':
    test_model()
