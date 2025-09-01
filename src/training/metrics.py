"""
Metrics for architectural style classification evaluation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


class ArchitecturalMetrics:
    """Metrics for architectural style classification."""
    
    def __init__(self, num_classes: int = 25, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Style_{i}" for i in range(num_classes)]
        
        # Initialize metric storage
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               probabilities: Optional[torch.Tensor] = None):
        """Update metrics with new predictions and targets."""
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        
        # Store predictions and targets
        self.predictions.extend(predictions)
        self.targets.extend(targets)
        if probabilities is not None:
            self.probabilities.extend(probabilities)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {}
        
        # Basic accuracy
        metrics['accuracy'] = accuracy_score(targets, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        # Macro averages
        metrics['macro_precision'] = np.mean(precision)
        metrics['macro_recall'] = np.mean(recall)
        metrics['macro_f1'] = np.mean(f1)
        
        # Weighted averages
        metrics['weighted_precision'] = np.average(precision, weights=support)
        metrics['weighted_recall'] = np.average(recall, weights=support)
        metrics['weighted_f1'] = np.average(f1, weights=support)
        
        # Per-class metrics - use actual number of classes from precision array
        actual_num_classes = len(precision)
        for i in range(actual_num_classes):
            metrics[f'precision_class_{i}'] = float(precision[i])
            metrics[f'recall_class_{i}'] = float(recall[i])
            metrics[f'f1_class_{i}'] = float(f1[i])
            metrics[f'support_class_{i}'] = float(support[i])
        
        # Confusion matrix (don't log this as it's a numpy array)
        cm = confusion_matrix(targets, predictions)
        # Store confusion matrix separately, don't include in logged metrics
        self.confusion_matrix = cm
        
        # ROC AUC (if probabilities are available)
        if self.probabilities:
            try:
                probabilities = np.array(self.probabilities)
                if probabilities.shape[1] == self.num_classes:
                    # One-vs-rest ROC AUC
                    auc_scores = []
                    actual_num_classes = min(self.num_classes, probabilities.shape[1])
                    for i in range(actual_num_classes):
                        try:
                            auc = roc_auc_score(
                                (targets == i).astype(int), 
                                probabilities[:, i]
                            )
                            auc_scores.append(auc)
                        except:
                            auc_scores.append(0.0)
                    
                    metrics['macro_auc'] = np.mean(auc_scores)
                    metrics['weighted_auc'] = np.average(auc_scores, weights=support[:actual_num_classes])
                    
                    # Per-class AUC
                    for i in range(actual_num_classes):
                        metrics[f'auc_class_{i}'] = float(auc_scores[i])
            except:
                pass
        
        return metrics
    
    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        if not self.predictions:
            return "No predictions available"
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        return classification_report(
            targets, predictions, 
            target_names=self.class_names,
            zero_division=0
        )
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        if not self.predictions:
            print("No predictions available for confusion matrix")
            return
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


class HierarchicalMetrics:
    """Metrics for hierarchical classification."""
    
    def __init__(self, num_broad_classes: int = 5, num_fine_classes: int = 25):
        self.num_broad_classes = num_broad_classes
        self.num_fine_classes = num_fine_classes
        
        # Style hierarchy mapping
        self.style_hierarchy = {
            0: [0, 1, 2, 3, 4],      # Ancient
            1: [5, 6, 7, 8, 9],      # Medieval  
            2: [10, 11, 12, 13, 14], # Renaissance
            3: [15, 16, 17, 18, 19], # Modern
            4: [20, 21, 22, 23, 24]  # Contemporary
        }
        
        self.reset()
    
    def reset(self):
        """Reset metrics."""
        self.broad_predictions = []
        self.broad_targets = []
        self.fine_predictions = []
        self.fine_targets = []
    
    def update(self, broad_pred: torch.Tensor, broad_target: torch.Tensor,
               fine_pred: torch.Tensor, fine_target: torch.Tensor):
        """Update hierarchical metrics."""
        if isinstance(broad_pred, torch.Tensor):
            broad_pred = broad_pred.cpu().numpy()
        if isinstance(broad_target, torch.Tensor):
            broad_target = broad_target.cpu().numpy()
        if isinstance(fine_pred, torch.Tensor):
            fine_pred = fine_pred.cpu().numpy()
        if isinstance(fine_target, torch.Tensor):
            fine_target = fine_target.cpu().numpy()
        
        self.broad_predictions.extend(broad_pred)
        self.broad_targets.extend(broad_target)
        self.fine_predictions.extend(fine_pred)
        self.fine_targets.extend(fine_target)
    
    def compute(self) -> Dict[str, float]:
        """Compute hierarchical metrics."""
        if not self.broad_predictions:
            return {}
        
        broad_pred = np.array(self.broad_predictions)
        broad_target = np.array(self.broad_targets)
        fine_pred = np.array(self.fine_predictions)
        fine_target = np.array(self.fine_targets)
        
        metrics = {}
        
        # Broad classification metrics
        metrics['broad_accuracy'] = accuracy_score(broad_target, broad_pred)
        broad_precision, broad_recall, broad_f1, _ = precision_recall_fscore_support(
            broad_target, broad_pred, average='macro', zero_division=0
        )
        metrics['broad_precision'] = broad_precision
        metrics['broad_recall'] = broad_recall
        metrics['broad_f1'] = broad_f1
        
        # Fine classification metrics
        metrics['fine_accuracy'] = accuracy_score(fine_target, fine_pred)
        fine_precision, fine_recall, fine_f1, _ = precision_recall_fscore_support(
            fine_target, fine_pred, average='macro', zero_division=0
        )
        metrics['fine_precision'] = fine_precision
        metrics['fine_recall'] = fine_recall
        metrics['fine_f1'] = fine_f1
        
        # Hierarchical consistency
        consistency_score = self._compute_hierarchical_consistency(
            broad_pred, broad_target, fine_pred, fine_target
        )
        metrics['hierarchical_consistency'] = consistency_score
        
        return metrics
    
    def _compute_hierarchical_consistency(self, broad_pred: np.ndarray, 
                                        broad_target: np.ndarray,
                                        fine_pred: np.ndarray, 
                                        fine_target: np.ndarray) -> float:
        """Compute hierarchical consistency score."""
        consistency_count = 0
        total_count = len(broad_pred)
        
        for i in range(total_count):
            # Check if fine-grained prediction is consistent with broad prediction
            broad_pred_class = broad_pred[i]
            fine_pred_class = fine_pred[i]
            
            # Get the broad class that the fine-grained class belongs to
            fine_broad_class = None
            for broad_class, fine_classes in self.style_hierarchy.items():
                if fine_pred_class in fine_classes:
                    fine_broad_class = broad_class
                    break
            
            # Check consistency
            if fine_broad_class == broad_pred_class:
                consistency_count += 1
        
        return consistency_count / total_count if total_count > 0 else 0.0


class StyleRelationshipMetrics:
    """Metrics for evaluating style relationship modeling."""
    
    def __init__(self, num_styles: int = 25):
        self.num_styles = num_styles
        self.reset()
    
    def reset(self):
        """Reset metrics."""
        self.style_embeddings = []
        self.style_predictions = []
        self.style_targets = []
    
    def update(self, embeddings: torch.Tensor, predictions: torch.Tensor, 
               targets: torch.Tensor):
        """Update style relationship metrics."""
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        self.style_embeddings.extend(embeddings)
        self.style_predictions.extend(predictions)
        self.style_targets.extend(targets)
    
    def compute(self) -> Dict[str, float]:
        """Compute style relationship metrics."""
        if not self.style_embeddings:
            return {}
        
        embeddings = np.array(self.style_embeddings)
        predictions = np.array(self.style_predictions)
        targets = np.array(self.style_targets)
        
        metrics = {}
        
        # Compute style similarity matrix
        similarity_matrix = self._compute_style_similarity_matrix(embeddings)
        
        # Style clustering quality
        clustering_score = self._compute_clustering_quality(similarity_matrix, targets)
        metrics['style_clustering_quality'] = clustering_score
        
        # Style relationship consistency
        relationship_consistency = self._compute_relationship_consistency(
            similarity_matrix, targets
        )
        metrics['style_relationship_consistency'] = relationship_consistency
        
        return metrics
    
    def _compute_style_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute style similarity matrix from embeddings."""
        # Normalize embeddings
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
        
        return similarity_matrix
    
    def _compute_clustering_quality(self, similarity_matrix: np.ndarray, 
                                  targets: np.ndarray) -> float:
        """Compute clustering quality score."""
        # Compute intra-class and inter-class similarities
        intra_class_similarities = []
        inter_class_similarities = []
        
        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                similarity = similarity_matrix[i, j]
                
                if targets[i] == targets[j]:
                    intra_class_similarities.append(similarity)
                else:
                    inter_class_similarities.append(similarity)
        
        if not intra_class_similarities or not inter_class_similarities:
            return 0.0
        
        # Compute clustering quality as the difference between intra and inter class similarities
        intra_mean = np.mean(intra_class_similarities)
        inter_mean = np.mean(inter_class_similarities)
        
        return intra_mean - inter_mean
    
    def _compute_relationship_consistency(self, similarity_matrix: np.ndarray,
                                        targets: np.ndarray) -> float:
        """Compute style relationship consistency."""
        # Define expected style relationships (simplified)
        periods = [
            list(range(0, 5)),    # Ancient
            list(range(5, 10)),   # Medieval
            list(range(10, 15)),  # Renaissance
            list(range(15, 20)),  # Modern
            list(range(20, 25))   # Contemporary
        ]
        
        consistency_scores = []
        
        for period in periods:
            period_similarities = []
            for i in period:
                for j in period:
                    if i != j:
                        period_similarities.append(similarity_matrix[i, j])
            
            if period_similarities:
                consistency_scores.append(np.mean(period_similarities))
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
