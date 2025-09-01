"""
Attention mechanisms for architectural style classification.
Provides interpretable attention for understanding model decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


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


class SpatialAttention(nn.Module):
    """Spatial attention for focusing on architectural elements."""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_attention = avg_out + max_out
        
        # Apply attention
        return x * channel_attention.view(b, c, 1, 1)


class StyleRelationshipAttention(nn.Module):
    """Attention mechanism for modeling style relationships."""
    
    def __init__(self, num_styles: int = 25, embedding_dim: int = 256):
        super().__init__()
        self.style_embeddings = nn.Parameter(torch.randn(num_styles, embedding_dim))
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Historical style relationships (simplified)
        self.style_relationships = self._initialize_style_relationships()
        
    def _initialize_style_relationships(self) -> torch.Tensor:
        """Initialize style relationship matrix based on architectural history."""
        # Simplified relationship matrix
        # 1 = related styles, 0 = unrelated
        relationships = torch.zeros(25, 25)
        
        # Ancient styles (0-4)
        relationships[0:5, 0:5] = 1
        
        # Medieval styles (5-9)
        relationships[5:10, 5:10] = 1
        
        # Renaissance styles (10-14)
        relationships[10:15, 10:15] = 1
        
        # Modern styles (15-19)
        relationships[15:20, 15:20] = 1
        
        # Contemporary styles (20-24)
        relationships[20:25, 20:25] = 1
        
        # Cross-period relationships (simplified)
        relationships[0:5, 5:10] = 0.3  # Ancient to Medieval
        relationships[5:10, 10:15] = 0.3  # Medieval to Renaissance
        relationships[10:15, 15:20] = 0.3  # Renaissance to Modern
        
        return relationships
    
    def forward(self, style_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply style relationship attention."""
        # Project to query, key, value
        queries = self.query_projection(style_features)
        keys = self.key_projection(self.style_embeddings)
        values = self.value_projection(self.style_embeddings)
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        
        # Apply style relationship mask
        relationship_mask = self.style_relationships.unsqueeze(0).expand(
            attention_scores.size(0), -1, -1
        )
        attention_scores = attention_scores * relationship_mask
        
        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended_features = torch.matmul(attention_weights, values)
        
        return attended_features, attention_weights


class ArchitecturalElementAttention(nn.Module):
    """Attention mechanism for architectural elements (arches, windows, columns, etc.)."""
    
    def __init__(self, num_elements: int = 10, feature_dim: int = 512):
        super().__init__()
        self.element_embeddings = nn.Parameter(torch.randn(num_elements, feature_dim))
        self.element_names = [
            'arches', 'windows', 'columns', 'domes', 'spires',
            'facades', 'ornaments', 'materials', 'proportions', 'symmetry'
        ]
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
        
    def forward(self, image_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply architectural element attention."""
        # Reshape features for attention
        b, c, h, w = image_features.size()
        spatial_features = image_features.view(b, c, h * w).transpose(1, 2)
        
        # Apply multi-head attention
        attended_features, attention_weights = self.attention(
            spatial_features, 
            self.element_embeddings.unsqueeze(0).expand(b, -1, -1),
            self.element_embeddings.unsqueeze(0).expand(b, -1, -1)
        )
        
        # Element importance scores
        element_scores = torch.mean(attention_weights, dim=1)  # Average over spatial positions
        
        return {
            'attended_features': attended_features,
            'attention_weights': attention_weights,
            'element_scores': element_scores,
            'element_names': self.element_names
        }


class CrossScaleAttention(nn.Module):
    """Cross-scale attention for combining global and local information."""
    
    def __init__(self, global_dim: int, local_dim: int, fusion_dim: int = 256):
        super().__init__()
        self.global_projection = nn.Linear(global_dim, fusion_dim)
        self.local_projection = nn.Linear(local_dim, fusion_dim)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.fusion_layer = nn.Linear(fusion_dim * 2, fusion_dim)
        
    def forward(self, global_features: torch.Tensor, 
                local_features: torch.Tensor) -> torch.Tensor:
        # Project features
        global_proj = self.global_projection(global_features).unsqueeze(1)
        local_proj = self.local_projection(local_features)
        
        # Cross attention
        attended_global, _ = self.cross_attention(
            global_proj, local_proj, local_proj
        )
        
        # Fuse features
        fused = torch.cat([
            attended_global.squeeze(1),
            global_features
        ], dim=1)
        
        return self.fusion_layer(fused)


class InterpretableAttention(nn.Module):
    """Interpretable attention mechanism with visualization capabilities."""
    
    def __init__(self, feature_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.attention_history = []
        self.feature_importance = []
        
    def forward(self, features: torch.Tensor, 
                return_attention: bool = True) -> Dict[str, torch.Tensor]:
        # Self-attention
        attended_features, attention_weights = self.attention(
            features, features, features
        )
        
        # Store for visualization
        if return_attention:
            self.attention_history.append(attention_weights.detach())
            
            # Compute feature importance
            feature_importance = torch.mean(attention_weights, dim=1)
            self.feature_importance.append(feature_importance.detach())
        
        return {
            'features': attended_features,
            'attention_weights': attention_weights,
            'feature_importance': torch.mean(attention_weights, dim=1) if return_attention else None
        }
    
    def get_attention_visualization(self) -> Dict[str, torch.Tensor]:
        """Get attention weights for visualization."""
        if not self.attention_history:
            return {}
        
        return {
            'attention_weights': self.attention_history[-1],
            'feature_importance': self.feature_importance[-1] if self.feature_importance else None,
            'attention_history': torch.stack(self.attention_history) if len(self.attention_history) > 1 else None
        }
    
    def clear_history(self):
        """Clear attention history."""
        self.attention_history.clear()
        self.feature_importance.clear()
