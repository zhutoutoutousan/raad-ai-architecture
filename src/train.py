#!/usr/bin/env python3 
"""
Main training script for Architectural Style Classification
Advanced Deep Learning Approach with Hierarchical Multi-Modal Architecture
"""

import os
import sys
import json
import argparse
from typing import Dict, Any
import torch
import pytorch_lightning as pl

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import HierarchicalArchitecturalClassifier, BaselineModels
from src.training.trainer import ArchitecturalTrainer, ExperimentRunner
from src.training.losses import CombinedLoss
from src.utils.config import load_config, save_config


def create_experiment_configs() -> Dict[str, Dict[str, Any]]:
    """Create different experiment configurations."""
    
    configs = {
        # Baseline experiments
        'baseline_resnet': {
            'experiment_name': 'baseline_resnet',
            'model_type': 'resnet',
            'num_classes': 25,
            'learning_rate': 1e-4,
            'max_epochs': 50,
            'batch_size': 32,
            'use_hierarchical_loss': False,
            'use_contrastive_loss': False,
            'use_style_relationship_loss': False,
            'use_wandb': False
        },
        
        'baseline_efficientnet': {
            'experiment_name': 'baseline_efficientnet',
            'model_type': 'efficientnet',
            'num_classes': 25,
            'learning_rate': 1e-4,
            'max_epochs': 50,
            'batch_size': 32,
            'use_hierarchical_loss': False,
            'use_contrastive_loss': False,
            'use_style_relationship_loss': False,
            'use_wandb': False
        },
        
        'baseline_vit': {
            'experiment_name': 'baseline_vit',
            'model_type': 'vit',
            'num_classes': 25,
            'learning_rate': 1e-4,
            'max_epochs': 50,
            'batch_size': 16,  # Smaller batch size for ViT
            'use_hierarchical_loss': False,
            'use_contrastive_loss': False,
            'use_style_relationship_loss': False,
            'use_wandb': False
        },
        
        # Hierarchical model experiments
        'hierarchical_basic': {
            'experiment_name': 'hierarchical_basic',
            'model_type': 'hierarchical',
            'num_classes': 25,
            'num_broad_classes': 5,
            'num_fine_classes': 25,
            'learning_rate': 1e-4,
            'max_epochs': 100,
            'batch_size': 16,
            'use_hierarchical_loss': True,
            'use_contrastive_loss': False,
            'use_style_relationship_loss': True,
            'curriculum_stages': [
                {'epochs': 20, 'classes': ['ancient', 'medieval', 'modern']},
                {'epochs': 80, 'classes': list(range(25))}
            ],
            'use_wandb': False
        },
        
        'hierarchical_contrastive': {
            'experiment_name': 'hierarchical_contrastive',
            'model_type': 'hierarchical',
            'num_classes': 25,
            'num_broad_classes': 5,
            'num_fine_classes': 25,
            'learning_rate': 1e-4,
            'max_epochs': 100,
            'batch_size': 16,
            'use_hierarchical_loss': True,
            'use_contrastive_loss': True,
            'use_style_relationship_loss': True,
            'curriculum_stages': [
                {'epochs': 20, 'classes': ['ancient', 'medieval', 'modern']},
                {'epochs': 80, 'classes': list(range(25))}
            ],
            'use_wandb': False
        },
        
        # Advanced experiments
        'hierarchical_advanced': {
            'experiment_name': 'hierarchical_advanced',
            'model_type': 'hierarchical',
            'num_classes': 25,
            'num_broad_classes': 5,
            'num_fine_classes': 25,
            'learning_rate': 5e-5,
            'max_epochs': 150,
            'batch_size': 16,
            'use_hierarchical_loss': True,
            'use_contrastive_loss': True,
            'use_style_relationship_loss': True,
            'use_mixed_precision': True,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 2,
            'curriculum_stages': [
                {'epochs': 30, 'classes': ['ancient', 'medieval', 'modern']},
                {'epochs': 60, 'classes': list(range(25))},
                {'epochs': 60, 'classes': list(range(25))}
            ],
            'use_wandb': True
        }
    }
    
    return configs


def run_single_experiment(config: Dict[str, Any], data_path: str = None):
    """Run a single experiment."""
    print(f"Starting experiment: {config['experiment_name']}")
    print(f"Model type: {config['model_type']}")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Initialize experiment runner
    runner = ExperimentRunner(config)
    
    # Run experiment
    try:
        trainer, pl_trainer = runner.run_experiment()
        print(f"Experiment {config['experiment_name']} completed successfully!")
        return trainer, pl_trainer
    except Exception as e:
        print(f"Experiment {config['experiment_name']} failed: {str(e)}")
        raise


def run_experiment_suite(experiment_names: list = None, data_path: str = None):
    """Run a suite of experiments."""
    configs = create_experiment_configs()
    
    if experiment_names is None:
        experiment_names = list(configs.keys())
    
    results = {}
    
    for exp_name in experiment_names:
        if exp_name not in configs:
            print(f"Warning: Experiment {exp_name} not found in configurations")
            continue
            
        print(f"\n{'='*50}")
        print(f"Running experiment: {exp_name}")
        print(f"{'='*50}")
        
        try:
            trainer, pl_trainer = run_single_experiment(configs[exp_name], data_path)
            results[exp_name] = {
                'status': 'success',
                'trainer': trainer,
                'pl_trainer': pl_trainer
            }
        except Exception as e:
            print(f"Experiment {exp_name} failed: {str(e)}")
            results[exp_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Save results summary
    save_experiment_results(results)
    
    return results


def save_experiment_results(results: Dict[str, Any]):
    """Save experiment results summary."""
    summary = {}
    
    for exp_name, result in results.items():
        if result['status'] == 'success':
            summary[exp_name] = {
                'status': 'success',
                'model_type': result['trainer'].model.__class__.__name__,
                'hyperparameters': result['trainer'].hparams
            }
        else:
            summary[exp_name] = {
                'status': 'failed',
                'error': result.get('error', 'Unknown error')
            }
    
    # Save to file
    os.makedirs('results', exist_ok=True)
    with open('results/experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nExperiment summary saved to results/experiment_summary.json")


def test_model_creation():
    """Test model creation to ensure everything works."""
    print("Testing model creation...")
    
    try:
        # Test hierarchical model
        hierarchical_model = HierarchicalArchitecturalClassifier()
        print(f"✓ Hierarchical model created successfully")
        print(f"  Parameters: {sum(p.numel() for p in hierarchical_model.parameters()):,}")
        
        # Test baseline models
        resnet_model = BaselineModels.resnet50()
        print(f"✓ ResNet-50 model created successfully")
        print(f"  Parameters: {sum(p.numel() for p in resnet_model.parameters()):,}")
        
        efficientnet_model = BaselineModels.efficientnet_b4()
        print(f"✓ EfficientNet-B4 model created successfully")
        print(f"  Parameters: {sum(p.numel() for p in efficientnet_model.parameters()):,}")
        
        vit_model = BaselineModels.vit_base()
        print(f"✓ ViT-Base model created successfully")
        print(f"  Parameters: {sum(p.numel() for p in vit_model.parameters()):,}")
        
        # Test loss functions
        combined_loss = CombinedLoss()
        print(f"✓ Combined loss function created successfully")
        
        print("\nAll model tests passed! ✓")
        return True
        
    except Exception as e:
        print(f"Model test failed: {str(e)}")
        return False

 
def main(): 
    """Main function."""
    parser = argparse.ArgumentParser(description='Architectural Style Classification Training')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Specific experiment to run')
    parser.add_argument('--suite', action='store_true',
                       help='Run the full experiment suite')
    parser.add_argument('--test', action='store_true',
                       help='Test model creation and setup')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to dataset')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom config file')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    pl.seed_everything(42)
    
    print("Architectural Style Classification Training")
    print("=" * 50)
    
    # Test mode
    if args.test:
        if test_model_creation():
            print("Setup test completed successfully!")
        else:
            print("Setup test failed!")
            return 1
    
    # Load custom config if provided
    if args.config:
        config = load_config(args.config)
        run_single_experiment(config, args.data_path)
        return 0
    
    # Run specific experiment
    if args.experiment:
        configs = create_experiment_configs()
        if args.experiment not in configs:
            print(f"Experiment '{args.experiment}' not found!")
            print(f"Available experiments: {list(configs.keys())}")
            return 1
        
        run_single_experiment(configs[args.experiment], args.data_path)
        return 0
    
    # Run experiment suite
    if args.suite:
        run_experiment_suite(data_path=args.data_path)
        return 0
    
    # Default: run basic hierarchical experiment
    print("No specific experiment specified. Running basic hierarchical experiment...")
    configs = create_experiment_configs()
    run_single_experiment(configs['hierarchical_basic'], args.data_path)
    
    return 0

 
if __name__ == "__main__": 
    exit(main())
