#!/usr/bin/env python
"""
Quick training script for the forgery detection model.
This trains on CASIA2 dataset and saves the best model weights.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.st import (
    SwinTransformerForForgery, 
    get_transforms, 
    CASIADataset, 
    SimpleImageDataset,
    evaluate_model
)
from tqdm import tqdm
import random
import numpy as np

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_quick():
    """Train the model on CASIA2 dataset"""
    set_seed(42)
    
    config = {
        'batch_size': 16,  # Reduced for memory
        'learning_rate': 3e-4,
        'num_epochs': 5,  # Quick training
        'patience': 2,
        'model_name': 'swin_tiny_patch4_window7_224',
        'save_dir': './models',
        'data_dir': '../data'  # Point to parent directory
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    casia2_path = Path(config['data_dir']) / 'CASIA2'
    
    if not casia2_path.exists():
        print(f"❌ Dataset not found at: {casia2_path.resolve()}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    train_transform, val_transform = get_transforms()
    
    print("Loading CASIA2 dataset...")
    
    ds2 = CASIADataset(str(casia2_path), transform=None, dataset_version='v2')
    samples = ds2.samples
    
    if len(samples) == 0:
        print("❌ No samples loaded!")
        return
    
    print(f"✓ Total samples: {len(samples)}")
    
    # Shuffle and split into train/val (80/20)
    random.shuffle(samples)
    train_size = int(0.8 * len(samples))
    train_samples = samples[:train_size]
    val_samples = samples[train_size:]
    
    print(f"✓ Training samples: {len(train_samples)}, Validation samples: {len(val_samples)}")
    
    # Create datasets with appropriate transforms
    train_dataset = SimpleImageDataset(train_samples, transform=train_transform)
    val_dataset = SimpleImageDataset(val_samples, transform=val_transform)
    
    # Use 0 workers on Windows
    num_workers = 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print("✓ Initializing model...")
    model = SwinTransformerForForgery(
        model_name=config['model_name'],
        num_classes=2,
        pretrained=True
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70 + "\n")
    
    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} - Training", leave=True)
        
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            labels = labels.long()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_samples:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct_predictions / total_samples
        
        val_acc, val_precision, val_recall, val_f1, val_loss = evaluate_model(model, val_loader, device)
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, os.path.join(config['save_dir'], 'best_model.pth'))
            
            print(f"  ✓ New best model saved with validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"  ⚠ No improvement (patience: {patience_counter}/{config['patience']})")
        
        if patience_counter >= config['patience']:
            print(f"\n✓ Early stopping triggered after {epoch+1} epochs")
            break
        
        print("-" * 70)

if __name__ == '__main__':
    print("="*70)
    print("Image Forgery Detection Model Training")
    print("="*70)
    print("\nDataset: CASIA2 (Authentic + Tampered images)")
    print("Model: Swin Transformer (swin_tiny_patch4_window7_224)")
    print("Training will use GPU if available\n")
    
    train_quick()
    
    print("\n" + "="*70)
    print("Training completed!")
    print("Model saved to: ./models/best_model.pth")
    print("="*70)
