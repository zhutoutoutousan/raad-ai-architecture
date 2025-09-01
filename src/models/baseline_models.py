"""
Baseline models for architectural style classification.
Includes original MLLR approach and modern deep learning baselines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import timm
from transformers import ViTForImageClassification, ViTConfig


class BaselineModels:
    """Collection of baseline models for comparison."""
    
    @staticmethod
    def resnet50(num_classes: int = 25, pretrained: bool = True) -> nn.Module:
        """ResNet-50 baseline."""
        return BaselineModelWrapper(
            timm.create_model('resnet50', pretrained=pretrained, num_classes=num_classes),
            'resnet50'
        )
    
    @staticmethod
    def efficientnet_b4(num_classes: int = 25, pretrained: bool = True) -> nn.Module:
        """EfficientNet-B4 baseline."""
        return BaselineModelWrapper(
            timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=num_classes),
            'efficientnet_b4'
        )
    
    @staticmethod
    def vit_base(num_classes: int = 25, pretrained: bool = True) -> nn.Module:
        """Vision Transformer baseline."""
        return BaselineModelWrapper(
            timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes),
            'vit_base'
        )
    
    @staticmethod
    def convnext_base(num_classes: int = 25, pretrained: bool = True) -> nn.Module:
        """ConvNeXt baseline."""
        return BaselineModelWrapper(
            timm.create_model('convnext_base', pretrained=pretrained, num_classes=num_classes),
            'convnext_base'
        )


class BaselineModelWrapper(nn.Module):
    """Wrapper for baseline models to return expected format."""
    
    def __init__(self, model: nn.Module, model_name: str):
        super().__init__()
        self.model = model
        self.model_name = model_name
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass that returns expected format."""
        logits = self.model(x)
        return {
            'fine_logits': logits,
            'model_name': self.model_name
        }


class MLLRBaseline(nn.Module):
    """
    Simplified MLLR (Multinomial Latent Logistic Regression) baseline.
    This is a simplified version of the original DPM-MLLR approach.
    """
    
    def __init__(self, input_dim: int = 2048, num_classes: int = 25, 
                 num_latent: int = 10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Latent variables (simulating DPM)
        self.latent_embeddings = nn.Parameter(torch.randn(num_latent, 512))
        
        # Class-specific parameters
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, 512))
        
        # MLLR parameters
        self.alpha = nn.Parameter(torch.ones(num_latent))
        self.beta = nn.Parameter(torch.zeros(num_classes))
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.feature_extractor(x)
        
        # Compute latent assignments
        latent_scores = torch.matmul(features, self.latent_embeddings.t())
        latent_probs = F.softmax(latent_scores, dim=1)
        
        # Compute class scores
        class_scores = torch.matmul(features, self.class_embeddings.t())
        
        # MLLR combination
        latent_contribution = torch.matmul(latent_probs, self.alpha.unsqueeze(1)).squeeze(1)
        final_scores = class_scores + self.beta.unsqueeze(0) + latent_contribution.unsqueeze(1)
        
        return {
            'logits': final_scores,
            'latent_probs': latent_probs,
            'class_scores': class_scores
        }


class MultiStyleDetector(nn.Module):
    """
    Multi-style detection model for buildings with mixed architectural styles.
    """
    
    def __init__(self, backbone_name: str = 'efficientnet_b4', num_classes: int = 25):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        
        # Multi-label classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Sigmoid()  # Multi-label output
        )
        
        # Style mixture detector
        self.mixture_detector = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone.forward_features(x)
        if isinstance(features, tuple):
            features = features[0]
        
        # Global average pooling
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        # Multi-label classification
        style_probs = self.classifier(pooled)
        
        # Mixture detection
        mixture_prob = self.mixture_detector(pooled)
        
        return {
            'style_probs': style_probs,
            'mixture_prob': mixture_prob,
            'is_mixture': mixture_prob > 0.5
        }


class ContrastiveArchitecturalModel(nn.Module):
    """
    Contrastive learning model for architectural style classification.
    """
    
    def __init__(self, backbone_name: str = 'resnet50', projection_dim: int = 128):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim),
            nn.L2Norm()
        )
        
        # Classification head
        self.classifier = nn.Linear(projection_dim, 25)
        
    def forward(self, x: torch.Tensor, mode: str = 'classify') -> Dict[str, torch.Tensor]:
        features = self.backbone.forward_features(x)
        if isinstance(features, tuple):
            features = features[0]
        
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        projected = self.projection_head(pooled)
        
        if mode == 'contrastive':
            return {'projections': projected}
        else:
            logits = self.classifier(projected)
            return {'logits': logits, 'projections': projected}


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for improved performance.
    """
    
    def __init__(self, model_names: List[str] = None, num_classes: int = 25):
        super().__init__()
        if model_names is None:
            model_names = ['resnet50', 'efficientnet_b4', 'vit_base']
        
        self.models = nn.ModuleList([
            timm.create_model(name, pretrained=True, num_classes=num_classes)
            for name in model_names
        ])
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(len(model_names)))
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_output = sum(w * out for w, out in zip(weights, outputs))
        
        return {
            'ensemble_logits': ensemble_output,
            'individual_outputs': outputs,
            'ensemble_weights': weights
        }


class PretrainedCNNLoader:
    """
    Utility class for loading pre-trained CNN models.
    """
    
    @staticmethod
    def load_pretrained_model(model_path: str, model_type: str = 'hierarchical') -> nn.Module:
        """Load a pre-trained model from checkpoint."""
        if model_type == 'hierarchical':
            model = HierarchicalArchitecturalClassifier()
        elif model_type == 'resnet':
            model = BaselineModels.resnet50()
        elif model_type == 'efficientnet':
            model = BaselineModels.efficientnet_b4()
        elif model_type == 'vit':
            model = BaselineModels.vit_base()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    @staticmethod
    def save_model(model: nn.Module, save_path: str, additional_info: Dict = None):
        """Save a model checkpoint."""
        checkpoint = {
            'state_dict': model.state_dict(),
            'model_type': model.__class__.__name__,
            'additional_info': additional_info or {}
        }
        torch.save(checkpoint, save_path)
