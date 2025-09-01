"""
Loss functions for architectural style classification.
Includes hierarchical loss, contrastive loss, and style relationship loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class HierarchicalLoss(nn.Module):
    """Hierarchical loss for ensuring consistency between broad and fine-grained classifications."""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.3):
        super().__init__()
        self.alpha = alpha  # Weight for broad classification loss
        self.beta = beta    # Weight for consistency loss
        
        # Style hierarchy mapping
        self.style_hierarchy = {
            0: [0, 1, 2, 3, 4],      # Ancient
            1: [5, 6, 7, 8, 9],      # Medieval  
            2: [10, 11, 12, 13, 14], # Renaissance
            3: [15, 16, 17, 18, 19], # Modern
            4: [20, 21, 22, 23, 24]  # Contemporary
        }
        
        self.broad_to_fine = self._create_broad_to_fine_mapping()
        
    def _create_broad_to_fine_mapping(self) -> Dict[int, int]:
        """Create mapping from fine-grained classes to broad classes."""
        mapping = {}
        for broad_class, fine_classes in self.style_hierarchy.items():
            for fine_class in fine_classes:
                mapping[fine_class] = broad_class
        return mapping
    
    def forward(self, broad_logits: torch.Tensor, fine_logits: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """Compute hierarchical loss."""
        batch_size = targets.size(0)
        
        # Convert fine-grained targets to broad targets
        broad_targets = torch.tensor([
            self.broad_to_fine[target.item()] for target in targets
        ], device=targets.device)
        
        # Broad classification loss
        broad_loss = F.cross_entropy(broad_logits, broad_targets)
        
        # Fine-grained classification loss
        fine_loss = F.cross_entropy(fine_logits, targets)
        
        # Consistency loss: ensure fine-grained predictions are consistent with broad predictions
        broad_probs = F.softmax(broad_logits, dim=1)
        fine_probs = F.softmax(fine_logits, dim=1)
        
        consistency_loss = self._compute_consistency_loss(broad_probs, fine_probs, targets)
        
        # Total hierarchical loss
        total_loss = fine_loss + self.alpha * broad_loss + self.beta * consistency_loss
        
        return total_loss
    
    def _compute_consistency_loss(self, broad_probs: torch.Tensor, 
                                 fine_probs: torch.Tensor, 
                                 targets: torch.Tensor) -> torch.Tensor:
        """Compute consistency loss between broad and fine predictions."""
        batch_size = targets.size(0)
        consistency_loss = 0.0
        
        for i in range(batch_size):
            target = targets[i].item()
            broad_class = self.broad_to_fine[target]
            
            # Get fine-grained probabilities for the correct broad category
            fine_in_broad = self.style_hierarchy[broad_class]
            fine_probs_in_broad = fine_probs[i, fine_in_broad]
            
            # Get broad probability for the correct category
            broad_prob = broad_probs[i, broad_class]
            
            # Consistency: fine-grained probabilities should sum to broad probability
            consistency_loss += F.mse_loss(
                fine_probs_in_broad.sum(), 
                broad_prob
            )
        
        return consistency_loss / batch_size


class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning better feature representations."""
    
    def __init__(self, temperature: float = 0.07, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, projections: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss."""
        # Normalize projections
        projections = F.normalize(projections, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.t()) / self.temperature
        
        # Create positive and negative masks
        batch_size = targets.size(0)
        targets_expanded = targets.unsqueeze(1).expand(-1, batch_size)
        positive_mask = (targets_expanded == targets_expanded.t()).float()
        negative_mask = 1 - positive_mask
        
        # Remove self-similarity
        positive_mask.fill_diagonal_(0)
        
        # Compute positive and negative similarities
        positive_similarities = similarity_matrix * positive_mask
        negative_similarities = similarity_matrix * negative_mask
        
        # Find hardest negative for each positive
        hardest_negative_similarities = negative_similarities.max(dim=1)[0]
        
        # Compute contrastive loss
        positive_similarities = positive_similarities.sum(dim=1)
        num_positives = positive_mask.sum(dim=1)
        
        # Avoid division by zero
        num_positives = torch.clamp(num_positives, min=1)
        positive_similarities = positive_similarities / num_positives
        
        # Contrastive loss
        loss = F.relu(self.margin - positive_similarities + hardest_negative_similarities)
        
        return loss.mean()


class StyleRelationshipLoss(nn.Module):
    """Loss for modeling relationships between architectural styles."""
    
    def __init__(self, relationship_weight: float = 0.1):
        super().__init__()
        self.relationship_weight = relationship_weight
        
        # Define style relationships (simplified)
        self.style_relationships = self._initialize_style_relationships()
        
    def _initialize_style_relationships(self) -> torch.Tensor:
        """Initialize style relationship matrix."""
        num_styles = 25
        relationships = torch.zeros(num_styles, num_styles)
        
        # Same period relationships
        periods = [
            list(range(0, 5)),    # Ancient
            list(range(5, 10)),   # Medieval
            list(range(10, 15)),  # Renaissance
            list(range(15, 20)),  # Modern
            list(range(20, 25))   # Contemporary
        ]
        
        for period in periods:
            for i in period:
                for j in period:
                    if i != j:
                        relationships[i, j] = 0.8  # High similarity within period
        
        # Cross-period relationships (evolutionary)
        cross_periods = [
            (list(range(0, 5)), list(range(5, 10))),      # Ancient -> Medieval
            (list(range(5, 10)), list(range(10, 15))),    # Medieval -> Renaissance
            (list(range(10, 15)), list(range(15, 20))),   # Renaissance -> Modern
            (list(range(15, 20)), list(range(20, 25)))    # Modern -> Contemporary
        ]
        
        for prev_period, next_period in cross_periods:
            for i in prev_period:
                for j in next_period:
                    relationships[i, j] = 0.3  # Medium similarity across periods
        
        return relationships
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute style relationship loss."""
        batch_size = targets.size(0)
        
        # Get predicted probabilities
        probs = F.softmax(logits, dim=1)
        
        # Compute relationship loss
        relationship_loss = 0.0
        
        for i in range(batch_size):
            target = targets[i].item()
            
            # Get relationship scores for the target style
            target_relationships = self.style_relationships[target]
            
            # Compute expected vs actual similarities
            for j in range(batch_size):
                if i != j:
                    other_target = targets[j].item()
                    expected_similarity = target_relationships[other_target]
                    
                    # Compute actual similarity between predictions
                    actual_similarity = F.cosine_similarity(
                        probs[i].unsqueeze(0), 
                        probs[j].unsqueeze(0)
                    )
                    
                    # Relationship loss
                    relationship_loss += F.mse_loss(
                        actual_similarity, 
                        torch.tensor(expected_similarity, device=logits.device)
                    )
        
        # Normalize by number of pairs
        num_pairs = batch_size * (batch_size - 1)
        relationship_loss = relationship_loss / num_pairs if num_pairs > 0 else 0
        
        return self.relationship_weight * relationship_loss


class MultiStyleLoss(nn.Module):
    """Loss for multi-style detection and classification."""
    
    def __init__(self, mixture_weight: float = 0.2):
        super().__init__()
        self.mixture_weight = mixture_weight
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, style_probs: torch.Tensor, mixture_prob: torch.Tensor,
                targets: torch.Tensor, is_mixture: torch.Tensor) -> torch.Tensor:
        """Compute multi-style loss."""
        batch_size = targets.size(0)
        
        # Style classification loss
        style_loss = self.ce_loss(style_probs, targets)
        
        # Mixture detection loss
        mixture_loss = self.bce_loss(mixture_prob, is_mixture.float())
        
        # Multi-label loss for mixtures
        multi_label_loss = 0.0
        for i in range(batch_size):
            if is_mixture[i]:
                # For mixtures, encourage multiple style predictions
                target_probs = style_probs[i]
                # Encourage diversity in predictions
                entropy = -torch.sum(target_probs * torch.log(target_probs + 1e-8))
                multi_label_loss += -entropy  # Maximize entropy for mixtures
        
        multi_label_loss = multi_label_loss / batch_size if batch_size > 0 else 0
        
        # Total loss
        total_loss = style_loss + self.mixture_weight * mixture_loss + 0.1 * multi_label_loss
        
        return total_loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization."""
    
    def __init__(self, smoothing: float = 0.1, num_classes: int = 25):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing loss."""
        # Create smoothed labels
        batch_size = targets.size(0)
        smoothed_labels = torch.zeros(batch_size, self.num_classes, device=logits.device)
        smoothed_labels.fill_(self.smoothing / (self.num_classes - 1))
        smoothed_labels.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        # Compute loss
        log_probs = F.log_softmax(logits, dim=1)
        loss = -torch.sum(smoothed_labels * log_probs, dim=1)
        return loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss function with multiple components."""
    
    def __init__(self, 
                 use_hierarchical: bool = True,
                 use_contrastive: bool = False,
                 use_style_relationship: bool = True,
                 use_focal: bool = False,
                 use_label_smoothing: bool = True,
                 weights: Dict[str, float] = None):
        super().__init__()
        
        self.use_hierarchical = use_hierarchical
        self.use_contrastive = use_contrastive
        self.use_style_relationship = use_style_relationship
        self.use_focal = use_focal
        self.use_label_smoothing = use_label_smoothing
        
        # Initialize loss functions
        self.hierarchical_loss = HierarchicalLoss() if use_hierarchical else None
        self.contrastive_loss = ContrastiveLoss() if use_contrastive else None
        self.style_relationship_loss = StyleRelationshipLoss() if use_style_relationship else None
        self.focal_loss = FocalLoss() if use_focal else None
        self.label_smoothing_loss = LabelSmoothingLoss() if use_label_smoothing else None
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Loss weights
        self.weights = weights or {
            'ce': 1.0,
            'hierarchical': 0.5,
            'contrastive': 0.1,
            'style_relationship': 0.1,
            'focal': 1.0,
            'label_smoothing': 1.0
        }
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute combined loss."""
        total_loss = 0.0
        loss_dict = {}
        
        # Classification loss
        if 'fine_logits' in outputs:
            if self.use_focal:
                ce_loss = self.focal_loss(outputs['fine_logits'], targets)
            elif self.use_label_smoothing:
                ce_loss = self.label_smoothing_loss(outputs['fine_logits'], targets)
            else:
                ce_loss = self.ce_loss(outputs['fine_logits'], targets)
            
            total_loss += self.weights['ce'] * ce_loss
            loss_dict['ce_loss'] = ce_loss
        
        # Hierarchical loss
        if self.use_hierarchical and self.hierarchical_loss and 'broad_logits' in outputs:
            hierarchical_loss = self.hierarchical_loss(
                outputs['broad_logits'], 
                outputs['fine_logits'], 
                targets
            )
            total_loss += self.weights['hierarchical'] * hierarchical_loss
            loss_dict['hierarchical_loss'] = hierarchical_loss
        
        # Style relationship loss
        if self.use_style_relationship and self.style_relationship_loss and 'fine_logits' in outputs:
            relationship_loss = self.style_relationship_loss(
                outputs['fine_logits'], 
                targets
            )
            total_loss += self.weights['style_relationship'] * relationship_loss
            loss_dict['style_relationship_loss'] = relationship_loss
        
        # Contrastive loss
        if self.use_contrastive and self.contrastive_loss and 'projections' in outputs:
            contrastive_loss = self.contrastive_loss(
                outputs['projections'], 
                targets
            )
            total_loss += self.weights['contrastive'] * contrastive_loss
            loss_dict['contrastive_loss'] = contrastive_loss
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict
