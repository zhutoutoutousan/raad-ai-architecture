"""
Simple but Powerful Advanced Pre-trained CNN Classifier
Uses EfficientNetV2 with advanced training techniques for architectural style classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, List, Tuple, Optional
import numpy as np


class SimpleAdvancedClassifier(nn.Module):
    """
    Simple but powerful classifier using EfficientNetV2 with advanced techniques:
    - EfficientNetV2 (state-of-the-art CNN)
    - Advanced feature extraction
    - Multi-scale pooling
    - Attention mechanism
    - Dropout and regularization
    """
    
    def __init__(self, num_classes: int = 25, dropout_rate: float = 0.3):
        super().__init__()
        
        # Pre-trained EfficientNetV2 backbone
        self.backbone = timm.create_model(
            'tf_efficientnetv2_m', 
            pretrained=True, 
            num_classes=0,
            global_pool=''
        )
        
        # Get feature dimensions
        self.feature_dim = self.backbone.num_features
        print(f"EfficientNetV2 feature dimension: {self.feature_dim}")
        
        # Multi-scale pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Feature enhancement
        self.feature_enhancement = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),  # *2 for avg + max pooling
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim // 4, num_classes)
        )
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features from backbone
        features = self.backbone.forward_features(x)
        
        # Multi-scale pooling
        avg_pooled = self.global_pool(features).flatten(1)
        max_pooled = self.max_pool(features).flatten(1)
        
        # Concatenate pooled features
        pooled_features = torch.cat([avg_pooled, max_pooled], dim=1)
        
        # Feature enhancement
        enhanced_features = self.feature_enhancement(pooled_features)
        
        # Apply attention
        attention_weights = self.attention(enhanced_features)
        attended_features = enhanced_features * attention_weights
        
        # Classification
        logits = self.classifier(attended_features)
        
        # Apply temperature scaling
        logits = logits / self.temperature
        
        return {
            'logits': logits,
            'features': attended_features,
            'attention_weights': attention_weights
        }


class AdvancedLossFunction(nn.Module):
    """Advanced loss function with label smoothing and focal loss."""
    
    def __init__(self, num_classes: int = 25, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
        # Loss functions
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = outputs['logits']
        
        # Cross entropy loss
        ce_loss = self.cross_entropy(logits, targets)
        
        # Focal loss for hard examples
        focal_loss = self.focal_loss(logits, targets)
        
        # Combine losses
        total_loss = 0.7 * ce_loss + 0.3 * focal_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'focal_loss': focal_loss
        }


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def create_simple_advanced_classifier(num_classes: int = 25) -> SimpleAdvancedClassifier:
    """Factory function to create the simple advanced classifier."""
    return SimpleAdvancedClassifier(num_classes=num_classes)


def create_advanced_loss(num_classes: int = 25) -> AdvancedLossFunction:
    """Factory function to create the advanced loss function."""
    return AdvancedLossFunction(num_classes=num_classes)
