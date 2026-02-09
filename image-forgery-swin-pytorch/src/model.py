import torch.nn as nn
import timm

class SwinTransformerForForgery(nn.Module):
    """Swin Transformer model for image forgery detection"""
    
    def __init__(self, model_name='swin_tiny_patch4_window7_224', num_classes=2, pretrained=True):
        super(SwinTransformerForForgery, self).__init__()
        
        # Load pretrained Swin Transformer
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Replace the classifier head
        if hasattr(self.backbone, 'head'):
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Linear(in_features, num_classes)
        elif hasattr(self.backbone, 'classifier'):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, num_classes)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        features = self.backbone.forward_features(x)
        
        # Global average pooling if needed
        if len(features.shape) > 2:
            features = features.mean(dim=[2, 3]) if len(features.shape) == 4 else features.mean(dim=1)
        
        features = self.dropout(features)
        logits = self.backbone.head(features)
        return logits