"""
Hierarchical Architectural Style Classifier
Combines global, local, and relationship modeling for architectural style classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import timm
from transformers import ViTModel, ViTConfig


class GlobalStyleBranch(nn.Module):
    """Global branch for overall architectural composition."""
    
    def __init__(self, model_name: str = 'efficientnet_b4', num_classes: int = 25):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        if isinstance(features, tuple):
            features = features[0]
        pooled = self.global_pool(features).flatten(1)
        return self.classifier(pooled)


class LocalDetailBranch(nn.Module):
    """Local branch for architectural elements using Vision Transformer."""
    
    def __init__(self, image_size: int = 224, patch_size: int = 16, 
                 num_classes: int = 25, dim: int = 768, depth: int = 12, 
                 heads: int = 12):
        super().__init__()
        self.vit_config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            hidden_size=dim,
            num_hidden_layers=depth,
            num_attention_heads=heads,
            intermediate_size=dim * 4
        )
        self.vit = ViTModel(self.vit_config)
        self.classifier = nn.Linear(dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.vit(x)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)


class RelationshipBranch(nn.Module):
    """Graph Neural Network for modeling architectural component relationships."""
    
    def __init__(self, num_classes: int = 25, hidden_dim: int = 256, 
                 num_layers: int = 3):
        super().__init__()
        # Use a simple CNN to extract features first
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Flatten and project to hidden dimension
        self.input_projection = nn.Linear(256, hidden_dim)
        self.gnn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features using CNN
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Simplified GNN implementation
        # In practice, you'd use torch_geometric for proper GNN
        x = self.input_projection(features)
        for layer in self.gnn_layers:
            x = F.relu(layer(x))
        return self.classifier(x)  # No need for mean since it's already pooled


class FeatureFusion(nn.Module):
    """Fuses features from different branches."""
    
    def __init__(self, global_dim: int, local_dim: int, relationship_dim: int, 
                 fusion_dim: int = 512):
        super().__init__()
        self.global_projection = nn.Linear(global_dim, fusion_dim)
        self.local_projection = nn.Linear(local_dim, fusion_dim)
        self.relationship_projection = nn.Linear(relationship_dim, fusion_dim)
        self.fusion_layer = nn.Linear(fusion_dim * 3, fusion_dim)
        
    def forward(self, global_feat: torch.Tensor, local_feat: torch.Tensor, 
                relationship_feat: torch.Tensor) -> torch.Tensor:
        global_proj = self.global_projection(global_feat)
        local_proj = self.local_projection(local_feat)
        relationship_proj = self.relationship_projection(relationship_feat)
        
        combined = torch.cat([global_proj, local_proj, relationship_proj], dim=1)
        return self.fusion_layer(combined)


class HierarchicalClassifier(nn.Module):
    """Hierarchical classifier for broad categories and fine-grained styles."""
    
    def __init__(self, input_dim: int = 512, broad_classes: int = 5, 
                 fine_classes: int = 25):
        super().__init__()
        self.broad_classifier = nn.Linear(input_dim, broad_classes)
        self.fine_classifier = nn.Linear(input_dim, fine_classes)
        
        # Style hierarchy mapping (simplified)
        self.style_hierarchy = {
            0: [0, 1, 2, 3, 4],      # Ancient
            1: [5, 6, 7, 8, 9],      # Medieval  
            2: [10, 11, 12, 13, 14], # Renaissance
            3: [15, 16, 17, 18, 19], # Modern
            4: [20, 21, 22, 23, 24]  # Contemporary
        }
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        broad_logits = self.broad_classifier(features)
        fine_logits = self.fine_classifier(features)
        
        # Apply hierarchical constraints
        constrained_fine_logits = self.apply_hierarchical_constraints(
            broad_logits, fine_logits
        )
        
        return {
            'broad_logits': broad_logits,
            'fine_logits': constrained_fine_logits,
            'broad_probs': F.softmax(broad_logits, dim=1),
            'fine_probs': F.softmax(constrained_fine_logits, dim=1)
        }
    
    def apply_hierarchical_constraints(self, broad_logits: torch.Tensor, 
                                     fine_logits: torch.Tensor) -> torch.Tensor:
        """Apply hierarchical constraints to ensure consistency."""
        broad_probs = F.softmax(broad_logits, dim=1)
        constrained_fine = fine_logits.clone()
        
        # Mask fine-grained logits based on broad category predictions
        for i in range(broad_probs.shape[1]):
            mask = (broad_probs[:, i] < 0.1).unsqueeze(1)  # Low confidence in broad category
            for fine_idx in self.style_hierarchy[i]:
                constrained_fine[:, fine_idx] = torch.where(
                    mask.squeeze(), 
                    constrained_fine[:, fine_idx] - 10.0,  # Penalize
                    constrained_fine[:, fine_idx]
                )
        
        return constrained_fine


class HierarchicalArchitecturalClassifier(nn.Module):
    """Main hierarchical architectural style classifier."""
    
    def __init__(self, num_broad_classes: int = 5, num_fine_classes: int = 25,
                 image_size: int = 224, use_pretrained: bool = True):
        super().__init__()
        
        # Initialize branches
        self.global_branch = GlobalStyleBranch(num_classes=num_fine_classes)
        self.local_branch = LocalDetailBranch(num_classes=num_fine_classes)
        self.relationship_branch = RelationshipBranch(num_classes=num_fine_classes)
        
        # Feature fusion
        self.feature_fusion = FeatureFusion(
            global_dim=num_fine_classes,
            local_dim=num_fine_classes, 
            relationship_dim=num_fine_classes
        )
        
        # Hierarchical classifier
        self.hierarchical_classifier = HierarchicalClassifier(
            input_dim=512,
            broad_classes=num_broad_classes,
            fine_classes=num_fine_classes
        )
        
        # Multi-scale attention
        self.attention = MultiScaleAttention(
            global_dim=num_fine_classes,
            local_dim=num_fine_classes
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features from each branch
        global_features = self.global_branch(x)
        local_features = self.local_branch(x)
        relationship_features = self.relationship_branch(x)
        
        # Apply attention mechanism
        attended_global, attended_local = self.attention(
            global_features, local_features
        )
        
        # Fuse features
        fused_features = self.feature_fusion(
            attended_global, attended_local, relationship_features
        )
        
        # Hierarchical classification
        outputs = self.hierarchical_classifier(fused_features)
        
        # Add attention weights for interpretability
        outputs['attention_weights'] = self.attention.get_attention_weights()
        
        return outputs
    
    def get_style_hierarchy(self) -> Dict[int, List[int]]:
        """Get the style hierarchy mapping."""
        return self.hierarchical_classifier.style_hierarchy


class MultiScaleAttention(nn.Module):
    """Multi-scale attention mechanism for interpretability."""
    
    def __init__(self, global_dim: int, local_dim: int, attention_dim: int = 256):
        super().__init__()
        self.global_projection = nn.Linear(global_dim, attention_dim)
        self.local_projection = nn.Linear(local_dim, attention_dim)
        self.attention_weights = nn.Parameter(torch.randn(attention_dim, attention_dim))
        self.attention_weights_history = []
        
    def forward(self, global_features: torch.Tensor, 
                local_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        global_proj = self.global_projection(global_features)
        local_proj = self.local_projection(local_features)
        
        # Compute attention weights
        attention_scores = torch.matmul(global_proj, self.attention_weights)
        attention_scores = torch.matmul(attention_scores, local_proj.transpose(-2, -1))
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Store attention weights for visualization
        self.attention_weights_history.append(attention_weights.detach())
        
        # Apply attention
        attended_global = torch.matmul(attention_weights, global_features)
        attended_local = torch.matmul(attention_weights.transpose(-2, -1), local_features)
        
        return attended_global, attended_local
    
    def get_attention_weights(self) -> torch.Tensor:
        """Get the latest attention weights for visualization."""
        if self.attention_weights_history:
            return self.attention_weights_history[-1]
        return torch.zeros(1, 1, 1)  # Placeholder
