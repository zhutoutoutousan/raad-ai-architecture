"""
Enhanced trainer for architectural style classification.
Includes advanced optimization techniques for better accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os
import json
from datetime import datetime

from .losses import HierarchicalLoss, ContrastiveLoss, StyleRelationshipLoss, FocalLoss, LabelSmoothingLoss
from .metrics import ArchitecturalMetrics
from .data_loader import EnhancedArchitecturalDataLoader


class EnhancedArchitecturalTrainer(pl.LightningModule):
    """Enhanced trainer for architectural style classification with advanced optimization."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=['model'])
        
        # Enhanced configuration
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.batch_size = config.get('batch_size', 8)
        self.num_classes = config.get('num_classes', 25)
        self.use_mixed_precision = config.get('use_mixed_precision', True)
        self.use_early_stopping = config.get('use_early_stopping', True)
        self.patience = config.get('patience', 15)
        self.gradient_clip_val = config.get('gradient_clip_val', 1.0)
        self.accumulate_grad_batches = config.get('accumulate_grad_batches', 2)
        
        # Enhanced loss functions
        self.use_focal_loss = config.get('use_focal_loss', True)
        self.use_label_smoothing = config.get('use_label_smoothing', True)
        self.use_contrastive_loss = config.get('use_contrastive_loss', True)
        
        # Initialize loss functions
        self._init_loss_functions()
        
        # Initialize metrics
        self.metrics = ArchitecturalMetrics(num_classes=self.num_classes)
        
        # Curriculum learning
        self.curriculum_stage = 0
        self.curriculum_classes_count = self.num_classes
        
        # Learning rate scheduling
        self.scheduler_step_size = config.get('scheduler_step_size', 10)
        self.scheduler_gamma = config.get('scheduler_gamma', 0.5)
        self.warmup_epochs = config.get('warmup_epochs', 5)
        
        # TensorBoard logger
        self.tensorboard_logger = TensorBoardLogger(
            save_dir='logs',
            name=f'architectural_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            version=None
        )
    
    def _init_loss_functions(self):
        """Initialize enhanced loss functions."""
        # Main classification loss
        if self.use_focal_loss:
            self.classification_loss = FocalLoss(
                alpha=1.0, 
                gamma=2.0, 
                num_classes=self.num_classes
            )
        elif self.use_label_smoothing:
            self.classification_loss = LabelSmoothingLoss(
                smoothing=0.1, 
                num_classes=self.num_classes
            )
        else:
            self.classification_loss = nn.CrossEntropyLoss()
        
        # Additional loss functions
        if self.use_contrastive_loss:
            self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        
        # Hierarchical loss for multi-scale features
        self.hierarchical_loss = HierarchicalLoss(
            num_classes=self.num_classes,
            hierarchy_weights=[1.0, 0.5, 0.25]
        )
        
        # Style relationship loss
        self.style_relationship_loss = StyleRelationshipLoss(
            num_classes=self.num_classes,
            temperature=0.1
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Enhanced training step with multiple loss components."""
        images, labels = batch
        
        # Forward pass
        outputs = self(images)
        
        # Extract logits
        if isinstance(outputs, dict):
            logits = outputs.get('fine_logits', outputs.get('logits'))
            features = outputs.get('features', None)
            hierarchical_outputs = outputs.get('hierarchical_outputs', None)
        else:
            logits = outputs
            features = None
            hierarchical_outputs = None
        
        # Calculate main classification loss
        if self.use_focal_loss or self.use_label_smoothing:
            main_loss = self.classification_loss(logits, labels)
        else:
            main_loss = self.classification_loss(logits, labels)
        
        # Calculate additional losses
        total_loss = main_loss
        loss_dict = {'main_loss': main_loss}
        
        # Hierarchical loss
        if hierarchical_outputs is not None:
            hierarchical_loss = self.hierarchical_loss(hierarchical_outputs, labels)
            total_loss += 0.3 * hierarchical_loss
            loss_dict['hierarchical_loss'] = hierarchical_loss
        
        # Contrastive loss
        if self.use_contrastive_loss and features is not None:
            contrastive_loss = self.contrastive_loss(features, labels)
            total_loss += 0.1 * contrastive_loss
            loss_dict['contrastive_loss'] = contrastive_loss
        
        # Style relationship loss
        style_loss = self.style_relationship_loss(logits, labels)
        total_loss += 0.05 * style_loss
        loss_dict['style_loss'] = style_loss
        
        # Calculate metrics
        with torch.no_grad():
            metrics = self.metrics.compute(logits, labels)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.log(f'train_{key}', float(value), prog_bar=True)
        
        # Log losses
        loss_dict['loss'] = total_loss
        for key, value in loss_dict.items():
            self.log(f'train_{key}', value, prog_bar=True)
        
        return loss_dict
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Enhanced validation step."""
        images, labels = batch
        
        # Forward pass
        outputs = self(images)
        
        # Extract logits
        if isinstance(outputs, dict):
            logits = outputs.get('fine_logits', outputs.get('logits'))
        else:
            logits = outputs
        
        # Calculate loss
        val_loss = self.classification_loss(logits, labels)
        
        # Calculate metrics
        metrics = self.metrics.compute(logits, labels)
        
        # Log validation metrics
        self.log('val_loss', val_loss, prog_bar=True)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log(f'val_{key}', float(value), prog_bar=True)
        
        return {'val_loss': val_loss, 'logits': logits, 'labels': labels}
    
    def on_validation_epoch_end(self) -> None:
        """Enhanced validation epoch end with detailed logging."""
        # Log curriculum learning progress
        self.log('curriculum_stage', float(self.curriculum_stage), prog_bar=True)
        self.log('curriculum_classes_count', float(self.curriculum_classes_count), prog_bar=True)
        
        # Log learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=True)
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Enhanced test step."""
        images, labels = batch
        
        # Forward pass
        outputs = self(images)
        
        # Extract logits
        if isinstance(outputs, dict):
            logits = outputs.get('fine_logits', outputs.get('logits'))
        else:
            logits = outputs
        
        # Calculate metrics
        metrics = self.metrics.compute(logits, labels)
        
        # Log test metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log(f'test_{key}', float(value), prog_bar=True)
        
        return {'logits': logits, 'labels': labels}
    
    def on_test_epoch_end(self) -> None:
        """Save test results."""
        # Save confusion matrix
        confusion_matrix = self.metrics.confusion_matrix
        if confusion_matrix is not None:
            np.save('results/confusion_matrix.npy', confusion_matrix.cpu().numpy())
        
        # Save detailed results
        results = {
            'model_name': self.model.__class__.__name__,
            'config': self.config,
            'test_metrics': {
                'accuracy': float(self.metrics.accuracy),
                'precision_macro': float(self.metrics.precision_macro),
                'recall_macro': float(self.metrics.recall_macro),
                'f1_macro': float(self.metrics.f1_macro),
                'precision_weighted': float(self.metrics.precision_weighted),
                'recall_weighted': float(self.metrics.recall_weighted),
                'f1_weighted': float(self.metrics.f1_weighted),
            }
        }
        
        # Save results
        os.makedirs('results', exist_ok=True)
        with open(f'results/{self.config.get("experiment_name", "test")}_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    def configure_optimizers(self):
        """Configure enhanced optimizers and schedulers."""
        # Enhanced optimizer with better parameters
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Enhanced learning rate scheduler
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the restart interval each time
            eta_min=1e-7  # Minimum learning rate
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def create_callbacks(self) -> List[pl.Callback]:
        """Create enhanced callbacks."""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath='models/checkpoints',
            filename=f'{self.config.get("experiment_name", "model")}-{{epoch:02d}}-{{val_loss:.4f}}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        )
        callbacks.append(checkpoint_callback)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)
        
        # Early stopping (optional)
        if self.use_early_stopping:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=self.patience,
                verbose=True
            )
            callbacks.append(early_stopping)
        
        return callbacks
    
    def create_data_loaders(self, data_path: str) -> Tuple[Any, Any, Any]:
        """Create enhanced data loaders."""
        # Enhanced data loader with better augmentation
        data_loader = EnhancedArchitecturalDataLoader(
            data_dir=data_path,
            batch_size=self.batch_size,
            num_workers=4,
            use_albumentations=True  # Use advanced augmentation
        )
        
        # Calculate sample sizes based on available data
        total_samples = len(data_loader.get_train_loader().dataset)
        train_samples = int(0.7 * total_samples)
        val_samples = max(1, int(0.15 * total_samples))
        test_samples = max(1, int(0.15 * total_samples))
        
        print(f"Data split: Train={train_samples}, Val={val_samples}, Test={test_samples}")
        
        train_loader = data_loader.get_train_loader(train_samples)
        val_loader = data_loader.get_val_loader(val_samples)
        test_loader = data_loader.get_test_loader(test_samples)
        
        return train_loader, val_loader, test_loader
    
    def update_curriculum(self, epoch: int):
        """Update curriculum learning stage."""
        # Progressive curriculum: start with fewer classes, gradually increase
        if epoch < 10:
            self.curriculum_stage = 0
            self.curriculum_classes_count = min(10, self.num_classes)
        elif epoch < 30:
            self.curriculum_stage = 1
            self.curriculum_classes_count = min(20, self.num_classes)
        else:
            self.curriculum_stage = 2
            self.curriculum_classes_count = self.num_classes
        
        # Update model for current curriculum stage
        self.update_model_for_stage()
    
    def update_model_for_stage(self):
        """Update model for current curriculum stage."""
        # This can be implemented to modify model behavior based on curriculum stage
        pass


class EnhancedExperimentRunner:
    """Enhanced experiment runner with advanced optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiment_name = config.get('experiment_name', 'enhanced_experiment')
        
    def run_experiment(self, model: nn.Module, data_path: str):
        """Run enhanced experiment."""
        print(f"Starting enhanced experiment: {self.experiment_name}")
        
        # Create enhanced trainer
        trainer = EnhancedArchitecturalTrainer(model, self.config)
        
        # Create data loaders
        train_loader, val_loader, test_loader = trainer.create_data_loaders(data_path)
        
        # Create callbacks
        callbacks = trainer.create_callbacks()
        
        # Create Lightning trainer
        lightning_trainer = pl.Trainer(
            max_epochs=self.config.get('epochs', 100),
            accelerator='auto',
            devices='auto',
            precision='16-mixed' if self.config.get('use_mixed_precision', True) else '32',
            gradient_clip_val=self.config.get('gradient_clip_val', 1.0),
            accumulate_grad_batches=self.config.get('accumulate_grad_batches', 2),
            callbacks=callbacks,
            logger=trainer.tensorboard_logger,
            log_every_n_steps=10,
            val_check_interval=0.5,  # Validate twice per epoch
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=True,
        )
        
        # Train the model
        lightning_trainer.fit(trainer, train_loader, val_loader)
        
        # Test the model
        lightning_trainer.test(trainer, test_loader)
        
        print(f"Enhanced experiment {self.experiment_name} completed successfully!")
        
        return trainer


# Keep backward compatibility
class ArchitecturalTrainer(EnhancedArchitecturalTrainer):
    """Backward compatibility wrapper."""
    pass

class ExperimentRunner(EnhancedExperimentRunner):
    """Backward compatibility wrapper."""
    pass
