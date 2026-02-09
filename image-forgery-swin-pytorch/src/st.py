# filepath: /image-forgery-swin-pytorch/image-forgery-swin-pytorch/src/st.py
# Complete Training Script for Image Forgery Detection using CASIA Datasets
# Using Swin Transformer with PyTorch

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import random

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class CASIADataset(Dataset):
    """Custom Dataset class for CASIA v1 and v2 datasets"""
    
    def __init__(self, root_dir, transform=None, dataset_version='v2'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.dataset_version = dataset_version
        self.samples = []
        
        # Load authentic and tampered images
        self._load_samples()
        
    def _load_samples(self):
        """Load image paths and labels"""
        if self.dataset_version == 'v1':
            authentic_dir = self.root_dir / 'Au'
            tampered_dir = self.root_dir / 'Sp'
        else:  # v2
            authentic_dir = self.root_dir / 'Au'
            tampered_dir = self.root_dir / 'Tp'
        
        if authentic_dir.exists():
            for img_path in authentic_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
                    self.samples.append((str(img_path), 0))
        
        if tampered_dir.exists():
            for img_path in tampered_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
                    self.samples.append((str(img_path), 1))
        
        print(f"Loaded {len(self.samples)} samples from CASIA {self.dataset_version}")
        
        authentic_count = sum(1 for _, label in self.samples if label == 0)
        tampered_count = sum(1 for _, label in self.samples if label == 1)
        print(f"Authentic: {authentic_count}, Tampered: {tampered_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            if self.transform:
                black_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
                return self.transform(black_image), label
            return None, label

class SimpleImageDataset(Dataset):
    """Dataset from a list of (path, label) tuples and a transform."""
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # return a black image if loading fails
            black_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            if self.transform:
                return self.transform(black_image), label
            return transforms.ToTensor()(black_image), label

def download_casia_datasets():
    """Download CASIA datasets (placeholder - you need to manually download)"""
    print("Note: CASIA datasets need to be manually downloaded from:")
    print("CASIA v1: http://forensics.idealtest.org/")
    print("CASIA v2: http://forensics.idealtest.org/")
    print("Please download and extract them to './data/CASIA1' and './data/CASIA2'")
    
    os.makedirs('./data/CASIA1', exist_ok=True)
    os.makedirs('./data/CASIA2', exist_ok=True)

class SwinTransformerForForgery(nn.Module):
    """Swin Transformer model for image forgery detection"""
    
    def __init__(self, model_name='swin_tiny_patch4_window7_224', num_classes=2, pretrained=True):
        super(SwinTransformerForForgery, self).__init__()
        
        # Create model with a classification head that outputs num_classes logits (shape: [N, num_classes])
        # letting timm configure the head correctly avoids returning spatial maps.
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Use the full backbone forward so internal pooling / token handling
        # is consistent with the head created above. The backbone returns logits [N, num_classes].
        return self.backbone(x); 

def get_transforms():
    """Define training and validation transforms"""
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def evaluate_model(model, dataloader, device):
    """Evaluate model and return metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            labels = labels.long()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    avg_loss = total_loss / len(dataloader)
    
    return accuracy, precision, recall, f1, avg_loss

def train_model():
    """Main training function"""
    
    config = {
        'batch_size': 32,
        'learning_rate': 3e-4,
        'num_epochs': 30,
        'patience': 3,
        'model_name': 'swin_tiny_patch4_window7_224',
        'save_dir': './models',
        'data_dir': './data'
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    casia1_path = Path(config['data_dir']) / 'CASIA1'
    casia2_path = Path(config['data_dir']) / 'CASIA2'
    
    if not casia1_path.exists() or not casia2_path.exists():
        download_casia_datasets()
        print("Please download datasets manually and run again.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_transform, val_transform = get_transforms()
    
    print("Loading datasets...")
    
    # Build flat list of (path,label) samples from available CASIA folders
    samples = []
    if casia1_path.exists():
        ds1 = CASIADataset(casia1_path, transform=None, dataset_version='v1')
        samples.extend(ds1.samples)
    if casia2_path.exists():
        ds2 = CASIADataset(casia2_path, transform=None, dataset_version='v2')
        samples.extend(ds2.samples)

    if len(samples) == 0:
        print("No datasets found!")
        return

    print(f"Total samples: {len(samples)}")

    # Shuffle and split into train/val (80/20)
    random.shuffle(samples)
    train_size = int(0.8 * len(samples))
    train_samples = samples[:train_size]
    val_samples = samples[train_size:]

    # Create datasets with appropriate transforms
    train_dataset = SimpleImageDataset(train_samples, transform=train_transform)
    val_dataset = SimpleImageDataset(val_samples, transform=val_transform)

    # Use a safe number of workers on Windows (0 or 4 depending on environment)
    num_workers = 0 if os.name == 'nt' else 4

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
    
    print("Initializing model...")
    model = SwinTransformerForForgery(
        model_name=config['model_name'],
        num_classes=2,
        pretrained=True
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    patience_counter = 0
    
    print("Starting training...")
    
    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} - Training")
        
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
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        
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
            
            print(f"New best model saved with validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        print("-" * 50)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], 'training_history.png'))
    plt.show()
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {os.path.join(config['save_dir'], 'best_model.pth')}")

def test_model():
    """Test the trained model"""
    config = {
        'batch_size': 32,
        'model_name': 'swin_tiny_patch4_window7_224',
        'save_dir': './models',
        'data_dir': './data'
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    _, val_transform = get_transforms()
    
    model = SwinTransformerForForgery(
        model_name=config['model_name'],
        num_classes=2,
        pretrained=False
    ).to(device)
    
    checkpoint = torch.load(os.path.join(config['save_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model with validation accuracy: {checkpoint['val_acc']:.4f}")
    
    # Test on validation set or create test set
    # Add your test evaluation code here

if __name__ == "__main__":
    print("Starting CASIA Image Forgery Detection Training")
    print("=" * 60)
    
    train_model()
    
    # Optionally test the model
    # test_model()