"""
Advanced Pre-trained CNN Classifier for Architectural Style Classification
Uses multiple state-of-the-art architectures with ensemble methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoImageProcessor, AutoModel
from typing import Dict, List, Tuple, Optional
import numpy as np


class AdvancedPretrainedClassifier(nn.Module):
    """
    Advanced pre-trained classifier using multiple architectures:
    - EfficientNetV2 (for general features)
    - ConvNeXt (for modern architectural features)
    - Swin Transformer (for hierarchical features)
    - Vision Transformer (for global attention)
    """
    
    def __init__(self, num_classes: int = 25, dropout_rate: float = 0.3):
        super().__init__()
        
        # Multiple pre-trained backbones
        self.efficientnet = timm.create_model(
            'tf_efficientnetv2_m', 
            pretrained=True, 
            num_classes=0,
            global_pool='avg'
        )
        
        self.convnext = timm.create_model(
            'convnext_base', 
            pretrained=True, 
            num_classes=0,
            global_pool='avg'
        )
        
        self.swin = timm.create_model(
            'swin_base_patch4_window7_224', 
            pretrained=True, 
            num_classes=0,
            global_pool='avg'
        )
        
        # Vision Transformer from HuggingFace
        self.vit_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.vit = AutoModel.from_pretrained('google/vit-base-patch16-224')
        
        # Feature dimensions
        self.efficientnet_dim = self.efficientnet.num_features
        self.convnext_dim = self.convnext.num_features
        self.swin_dim = self.swin.num_features
        self.vit_dim = 768  # ViT base hidden size
        
        # Print feature dimensions for debugging
        print(f"Feature dimensions:")
        print(f"  EfficientNet: {self.efficientnet_dim}")
        print(f"  ConvNeXt: {self.convnext_dim}")
        print(f"  Swin: {self.swin_dim}")
        print(f"  ViT: {self.vit_dim}")
        
        # Feature fusion layers
        total_features = self.efficientnet_dim + self.convnext_dim + self.swin_dim + self.vit_dim
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Multi-scale attention
        self.attention = MultiScaleAttention(
            efficientnet_dim=self.efficientnet_dim,
            convnext_dim=self.convnext_dim,
            swin_dim=self.swin_dim,
            vit_dim=self.vit_dim
        )
        
        # Final classifier with multiple heads
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Auxiliary classifiers for each backbone
        self.aux_efficientnet = nn.Linear(self.efficientnet_dim, num_classes)
        self.aux_convnext = nn.Linear(self.convnext_dim, num_classes)
        self.aux_swin = nn.Linear(self.swin_dim, num_classes)
        self.aux_vit = nn.Linear(self.vit_dim, num_classes)
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features from each backbone
        efficientnet_features = self.efficientnet.forward_features(x)
        if isinstance(efficientnet_features, tuple):
            efficientnet_features = efficientnet_features[0]
        efficientnet_features = F.adaptive_avg_pool2d(efficientnet_features, 1).flatten(1)
        
        convnext_features = self.convnext.forward_features(x)
        if isinstance(convnext_features, tuple):
            convnext_features = convnext_features[0]
        convnext_features = F.adaptive_avg_pool2d(convnext_features, 1).flatten(1)
        
        swin_features = self.swin.forward_features(x)
        if isinstance(swin_features, tuple):
            swin_features = swin_features[0]
        swin_features = F.adaptive_avg_pool2d(swin_features, 1).flatten(1)
        
        # ViT features (need to process differently)
        vit_features = self._extract_vit_features(x)
        
        # Apply attention mechanism
        attended_features = self.attention(
            efficientnet_features, convnext_features, swin_features, vit_features
        )
        
        # Concatenate all features
        combined_features = torch.cat([
            efficientnet_features, convnext_features, swin_features, vit_features
        ], dim=1)
        
        # Feature fusion
        fused_features = self.feature_fusion(combined_features)
        
        # Main classifier
        main_logits = self.classifier(fused_features)
        
        # Auxiliary classifiers
        aux_efficientnet_logits = self.aux_efficientnet(efficientnet_features)
        aux_convnext_logits = self.aux_convnext(convnext_features)
        aux_swin_logits = self.aux_swin(swin_features)
        aux_vit_logits = self.aux_vit(vit_features)
        
        # Apply temperature scaling
        main_logits = main_logits / self.temperature
        
        return {
            'logits': main_logits,
            'aux_efficientnet': aux_efficientnet_logits,
            'aux_convnext': aux_convnext_logits,
            'aux_swin': aux_swin_logits,
            'aux_vit': aux_vit_logits,
            'features': fused_features,
            'attended_features': attended_features
        }
    
    def _extract_vit_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from Vision Transformer."""
        # Convert to PIL-like format for ViT
        # ViT expects normalized images in [0, 1] range
        x_normalized = x / 255.0
        
        # Use the CLS token output as features
        with torch.no_grad():
            outputs = self.vit(pixel_values=x_normalized)
            # Get the CLS token (first token)
            cls_output = outputs.last_hidden_state[:, 0, :]
        
        return cls_output


class MultiScaleAttention(nn.Module):
    """Multi-scale attention mechanism for feature fusion."""
    
    def __init__(self, efficientnet_dim: int, convnext_dim: int, swin_dim: int, vit_dim: int):
        super().__init__()
        
        # Project all features to a common dimension
        self.common_dim = 512
        
        # Projection layers to common dimension
        self.efficientnet_projection = nn.Linear(efficientnet_dim, self.common_dim)
        self.convnext_projection = nn.Linear(convnext_dim, self.common_dim)
        self.swin_projection = nn.Linear(swin_dim, self.common_dim)
        self.vit_projection = nn.Linear(vit_dim, self.common_dim)
        
        # Attention weights for each feature type
        self.efficientnet_attention = nn.Linear(self.common_dim, 1)
        self.convnext_attention = nn.Linear(self.common_dim, 1)
        self.swin_attention = nn.Linear(self.common_dim, 1)
        self.vit_attention = nn.Linear(self.common_dim, 1)
        
    def forward(self, efficientnet_features: torch.Tensor, convnext_features: torch.Tensor,
                swin_features: torch.Tensor, vit_features: torch.Tensor) -> torch.Tensor:
        
        # Project all features to common dimension
        efficientnet_proj = self.efficientnet_projection(efficientnet_features)
        convnext_proj = self.convnext_projection(convnext_features)
        swin_proj = self.swin_projection(swin_features)
        vit_proj = self.vit_projection(vit_features)
        
        # Calculate attention weights
        efficientnet_attn = torch.sigmoid(self.efficientnet_attention(efficientnet_proj))
        convnext_attn = torch.sigmoid(self.convnext_attention(convnext_proj))
        swin_attn = torch.sigmoid(self.swin_attention(swin_proj))
        vit_attn = torch.sigmoid(self.vit_attention(vit_proj))
        
        # Weighted features
        weighted_efficientnet = efficientnet_proj * efficientnet_attn
        weighted_convnext = convnext_proj * convnext_attn
        weighted_swin = swin_proj * swin_attn
        weighted_vit = vit_proj * vit_attn
        
        # Combine weighted features
        attended_features = (
            weighted_efficientnet + weighted_convnext + weighted_swin + weighted_vit
        ) / 4.0
        
        return attended_features


class AdvancedLossFunction(nn.Module):
    """Advanced loss function combining multiple loss types."""
    
    def __init__(self, num_classes: int = 25, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3):
        super().__init__()
        self.alpha = alpha  # Main loss weight
        self.beta = beta    # Auxiliary loss weight
        self.gamma = gamma  # Focal loss weight
        
        # Loss functions
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        self.center_loss = CenterLoss(num_classes=num_classes, feat_dim=512)
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        main_logits = outputs['logits']
        aux_logits = [
            outputs['aux_efficientnet'],
            outputs['aux_convnext'],
            outputs['aux_swin'],
            outputs['aux_vit']
        ]
        features = outputs['features']
        
        # Main classification loss
        main_loss = self.cross_entropy(main_logits, targets)
        
        # Auxiliary losses
        aux_losses = []
        for aux_logit in aux_logits:
            aux_loss = self.cross_entropy(aux_logit, targets)
            aux_losses.append(aux_loss)
        aux_loss = torch.mean(torch.stack(aux_losses))
        
        # Focal loss for hard examples
        focal_loss = self.focal_loss(main_logits, targets)
        
        # Center loss for feature learning
        center_loss = self.center_loss(features, targets)
        
        # Total loss
        total_loss = (
            self.alpha * main_loss +
            self.beta * aux_loss +
            self.gamma * focal_loss +
            0.1 * center_loss
        )
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'aux_loss': aux_loss,
            'focal_loss': focal_loss,
            'center_loss': center_loss
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


class CenterLoss(nn.Module):
    """Center Loss for learning discriminative features."""
    
    def __init__(self, num_classes: int, feat_dim: int, device: str = 'cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        
    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        centers_batch = self.centers.index_select(0, targets)
        return F.mse_loss(features, centers_batch)


def create_advanced_classifier(num_classes: int = 25) -> AdvancedPretrainedClassifier:
    """Factory function to create the advanced classifier."""
    return AdvancedPretrainedClassifier(num_classes=num_classes)


def create_advanced_loss(num_classes: int = 25) -> AdvancedLossFunction:
    """Factory function to create the advanced loss function."""
    return AdvancedLossFunction(num_classes=num_classes)
