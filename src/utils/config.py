"""
Configuration utilities for architectural style classification experiments.
"""

import json
import os
from typing import Dict, Any, Optional
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config = json.load(f)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to JSON or YAML file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.endswith('.json'):
            json.dump(config, f, indent=2, default=str)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters."""
    required_fields = ['experiment_name', 'model_type', 'num_classes']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate model type
    valid_model_types = ['hierarchical', 'resnet', 'efficientnet', 'vit']
    if config['model_type'] not in valid_model_types:
        raise ValueError(f"Invalid model_type: {config['model_type']}. Must be one of {valid_model_types}")
    
    # Validate numeric fields
    numeric_fields = ['num_classes', 'learning_rate', 'max_epochs', 'batch_size']
    for field in numeric_fields:
        if field in config and not isinstance(config[field], (int, float)):
            raise ValueError(f"Field {field} must be numeric")
    
    return True


def create_default_config() -> Dict[str, Any]:
    """Create a default configuration."""
    return {
        'experiment_name': 'architectural_classification',
        'model_type': 'hierarchical',
        'num_classes': 25,
        'num_broad_classes': 5,
        'num_fine_classes': 25,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'max_epochs': 100,
        'batch_size': 16,
        'use_hierarchical_loss': True,
        'use_contrastive_loss': False,
        'use_style_relationship_loss': True,
        'use_mixed_precision': False,
        'gradient_clip_val': 1.0,
        'accumulate_grad_batches': 1,
        'use_wandb': False,
        'curriculum_stages': [
            {'epochs': 20, 'classes': ['ancient', 'medieval', 'modern']},
            {'epochs': 80, 'classes': list(range(25))}
        ]
    }
