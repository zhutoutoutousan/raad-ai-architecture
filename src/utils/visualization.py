"""
Visualization utilities for architectural style classification.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

def plot_attention_weights(attention_weights: torch.Tensor, 
                          image: torch.Tensor,
                          title: str = "Attention Weights",
                          save_path: Optional[str] = None) -> None:
    """
    Plot attention weights overlaid on the image.
    
    Args:
        attention_weights: Attention weights tensor [H, W]
        image: Input image tensor [C, H, W]
        title: Plot title
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image
    img_np = image.permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    ax1.imshow(img_np)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Plot attention weights
    attention_np = attention_weights.numpy()
    im = ax2.imshow(attention_np, cmap='hot', alpha=0.7)
    ax2.imshow(img_np, alpha=0.3)
    ax2.set_title(title)
    ax2.axis('off')
    
    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_training_curves(train_losses: List[float], 
                        val_losses: List[float],
                        train_accuracies: Optional[List[float]] = None,
                        val_accuracies: Optional[List[float]] = None,
                        save_path: Optional[str] = None) -> None:
    """
    Plot training and validation curves.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_accuracies: Training accuracies per epoch
        val_accuracies: Validation accuracies per epoch
        save_path: Path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    if train_accuracies and val_accuracies:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot losses
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix(confusion_matrix: np.ndarray,
                         class_names: List[str],
                         title: str = "Confusion Matrix",
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_class_distribution(class_counts: Dict[str, int],
                           title: str = "Class Distribution",
                           save_path: Optional[str] = None) -> None:
    """
    Plot class distribution.
    
    Args:
        class_counts: Dictionary mapping class names to counts
        title: Plot title
        save_path: Path to save the plot
    """
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, counts)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_feature_maps(feature_maps: torch.Tensor,
                     title: str = "Feature Maps",
                     num_maps: int = 16,
                     save_path: Optional[str] = None) -> None:
    """
    Plot feature maps from a convolutional layer.
    
    Args:
        feature_maps: Feature maps tensor [B, C, H, W]
        title: Plot title
        num_maps: Number of feature maps to display
        save_path: Path to save the plot
    """
    # Take first batch and select first num_maps channels
    maps = feature_maps[0, :num_maps].detach().cpu()
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_maps)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_maps):
        if i < maps.shape[0]:
            im = axes[i].imshow(maps[i], cmap='viridis')
            axes[i].set_title(f'Feature Map {i+1}')
            axes[i].axis('off')
        else:
            axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_learning_rate_schedule(learning_rates: List[float],
                               steps: List[int],
                               title: str = "Learning Rate Schedule",
                               save_path: Optional[str] = None) -> None:
    """
    Plot learning rate schedule.
    
    Args:
        learning_rates: List of learning rates
        steps: List of step numbers
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(steps, learning_rates, 'b-', linewidth=2)
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
