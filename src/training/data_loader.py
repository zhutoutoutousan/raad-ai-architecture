"""
Enhanced data loader for architectural style classification.
Includes advanced augmentation and better data handling.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os
from PIL import Image
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


class EnhancedArchitecturalDataset(Dataset):
    """Enhanced dataset for architectural style classification with advanced augmentation."""
    
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None, 
                 split: str = 'train', num_samples: Optional[int] = None, use_albumentations: bool = True):
        self.data_dir = data_dir
        self.split = split
        self.use_albumentations = use_albumentations
        
        # Use enhanced transforms if albumentations is available
        if use_albumentations:
            self.transform = transform or self._get_enhanced_transform()
        else:
            self.transform = transform or self._get_default_transform()
        
        # Load data paths and labels
        self.data_paths, self.labels = self._load_data()
        
        # Limit samples if specified
        if num_samples and len(self.data_paths) > 0:
            # Ensure we don't try to sample more than available
            actual_samples = min(num_samples, len(self.data_paths))
            indices = random.sample(range(len(self.data_paths)), actual_samples)
            self.data_paths = [self.data_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
    
    def _load_data(self) -> Tuple[List[str], List[int]]:
        """Load data paths and labels."""
        data_paths = []
        labels = []
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory {self.data_dir} does not exist. Using sample data.")
            return self._generate_sample_data()
        
        # First try to load from directory structure directly in data_dir (real data)
        real_data_found = False
        for class_idx in range(25):  # 25 architectural styles
            class_dir = os.path.join(self.data_dir, str(class_idx))
            if os.path.exists(class_dir):
                real_data_found = True
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        data_paths.append(os.path.join(class_dir, filename))
                        labels.append(class_idx)
        
        if real_data_found:
            print(f"Loading real data from directory: {self.data_dir}")
            return data_paths, labels
        
        # Fallback to sample_data subdirectory if no real data found
        sample_data_dir = os.path.join(self.data_dir, 'sample_data')
        if os.path.exists(sample_data_dir):
            print(f"Loading data from sample_data directory: {sample_data_dir}")
            # Load from sample_data directory structure
            for class_idx in range(25):  # 25 architectural styles
                class_dir = os.path.join(sample_data_dir, str(class_idx))
                if os.path.exists(class_dir):
                    for filename in os.listdir(class_dir):
                        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                            data_paths.append(os.path.join(class_dir, filename))
                            labels.append(class_idx)
        
        return data_paths, labels
    
    def _get_enhanced_transform(self) -> A.Compose:
        """Get enhanced transforms using Albumentations."""
        if self.split == 'train':
            return A.Compose([
                A.Resize(256, 256),
                A.RandomCrop(224, 224, p=0.8),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=0.3),
                    A.MedianBlur(blur_limit=3, p=0.3),
                    A.Blur(blur_limit=3, p=0.3),
                ], p=0.2),
                A.OneOf([
                    A.CLAHE(clip_limit=2, p=0.3),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                ], p=0.5),
                A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                    A.ISONoise(color_shift=(0.01, 0.05), p=0.3),
                ], p=0.2),
                A.OneOf([
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                    A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=0.3),
                ], p=0.2),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default transforms for architectural images."""
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _generate_sample_data(self) -> Tuple[List[str], List[int]]:
        """Generate sample data for testing."""
        print("Generating sample data for testing...")
        
        # Create sample data directory
        sample_dir = os.path.join(self.data_dir, 'sample_data')
        os.makedirs(sample_dir, exist_ok=True)
        
        data_paths = []
        labels = []
        
        # Generate sample images for each class
        for class_idx in range(25):
            class_dir = os.path.join(sample_dir, str(class_idx))
            os.makedirs(class_dir, exist_ok=True)
            
            # Generate 20 sample images per class (increased from 10)
            for i in range(20):
                # Create a simple colored image as placeholder
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                
                # Add some class-specific patterns
                if class_idx < 5:  # Ancient styles
                    img_array[:, :, 0] = np.random.randint(100, 200)  # Reddish
                elif class_idx < 10:  # Medieval styles
                    img_array[:, :, 1] = np.random.randint(100, 200)  # Greenish
                elif class_idx < 15:  # Renaissance styles
                    img_array[:, :, 2] = np.random.randint(100, 200)  # Bluish
                elif class_idx < 20:  # Modern styles
                    img_array[:, :, :] = np.random.randint(150, 255)  # Bright
                else:  # Contemporary styles
                    img_array[:, :, :] = np.random.randint(0, 100)    # Dark
                
                # Save image
                img = Image.fromarray(img_array)
                img_path = os.path.join(class_dir, f'sample_{i}.jpg')
                img.save(img_path)
                
                data_paths.append(img_path)
                labels.append(class_idx)
        
        print(f"Generated {len(data_paths)} sample images")
        return data_paths, labels
    
    def __len__(self) -> int:
        return len(self.data_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.data_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # If image loading fails, create a random image
            image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Apply transforms
        if self.use_albumentations and isinstance(self.transform, A.Compose):
            # Convert PIL image to numpy array for Albumentations
            image_np = np.array(image)
            transformed = self.transform(image=image_np)
            image = transformed['image']
        else:
            # Use torchvision transforms
            if self.transform:
                image = self.transform(image)
        
        return image, label


class EnhancedArchitecturalDataLoader:
    """Enhanced data loader factory for architectural style classification."""
    
    def __init__(self, data_dir: str, batch_size: int = 16, num_workers: int = 4, use_albumentations: bool = True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_albumentations = use_albumentations
        
        # Define transforms
        self.train_transform = self._get_train_transform()
        self.val_transform = self._get_val_transform()
        self.test_transform = self._get_test_transform()
    
    def _get_train_transform(self):
        """Get training transforms with advanced augmentation."""
        if self.use_albumentations:
            return A.Compose([
                A.Resize(256, 256),
                A.RandomCrop(224, 224, p=0.8),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=0.3),
                    A.MedianBlur(blur_limit=3, p=0.3),
                    A.Blur(blur_limit=3, p=0.3),
                ], p=0.2),
                A.OneOf([
                    A.CLAHE(clip_limit=2, p=0.3),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                ], p=0.5),
                A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                    A.ISONoise(color_shift=(0.01, 0.05), p=0.3),
                ], p=0.2),
                A.OneOf([
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                    A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=0.3),
                ], p=0.2),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _get_val_transform(self):
        """Get validation transforms."""
        if self.use_albumentations:
            return A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _get_test_transform(self):
        """Get test transforms."""
        return self._get_val_transform()
    
    def get_train_loader(self, num_samples: Optional[int] = None) -> DataLoader:
        """Get training data loader."""
        dataset = EnhancedArchitecturalDataset(
            self.data_dir, 
            transform=self.train_transform, 
            split='train',
            num_samples=num_samples,
            use_albumentations=self.use_albumentations
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True  # Drop incomplete batches for better training
        )
    
    def get_val_loader(self, num_samples: Optional[int] = None) -> DataLoader:
        """Get validation data loader."""
        dataset = EnhancedArchitecturalDataset(
            self.data_dir, 
            transform=self.val_transform, 
            split='val',
            num_samples=num_samples,
            use_albumentations=self.use_albumentations
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_test_loader(self, num_samples: Optional[int] = None) -> DataLoader:
        """Get test data loader."""
        dataset = EnhancedArchitecturalDataset(
            self.data_dir, 
            transform=self.test_transform, 
            split='test',
            num_samples=num_samples,
            use_albumentations=self.use_albumentations
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_all_loaders(self, num_samples: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get all data loaders."""
        train_loader = self.get_train_loader(num_samples)
        val_loader = self.get_val_loader(num_samples)
        test_loader = self.get_test_loader(num_samples)
        
        return train_loader, val_loader, test_loader


# Keep the original classes for backward compatibility
class ArchitecturalDataset(EnhancedArchitecturalDataset):
    """Backward compatibility wrapper."""
    pass

class ArchitecturalDataLoader(EnhancedArchitecturalDataLoader):
    """Backward compatibility wrapper."""
    pass


class SampleDataGenerator:
    """Generate sample data for testing and development."""
    
    def __init__(self, output_dir: str = 'data/sample'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_sample_dataset(self, num_classes: int = 25, samples_per_class: int = 100):
        """Generate a complete sample dataset."""
        print(f"Generating sample dataset with {num_classes} classes and {samples_per_class} samples per class...")
        
        for class_idx in range(num_classes):
            class_dir = os.path.join(self.output_dir, str(class_idx))
            os.makedirs(class_dir, exist_ok=True)
            
            for sample_idx in range(samples_per_class):
                # Generate sample image
                img_array = self._generate_sample_image(class_idx)
                
                # Save image
                img = Image.fromarray(img_array)
                img_path = os.path.join(class_dir, f'sample_{sample_idx:03d}.jpg')
                img.save(img_path)
        
        print(f"Sample dataset generated in {self.output_dir}")
        print(f"Total images: {num_classes * samples_per_class}")
    
    def _generate_sample_image(self, class_idx: int) -> np.ndarray:
        """Generate a sample image for a specific class."""
        # Base image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Add class-specific characteristics
        if class_idx < 5:  # Ancient styles (Greek, Roman, etc.)
            # Add columns and arches pattern
            img_array = self._add_ancient_patterns(img_array)
        elif class_idx < 10:  # Medieval styles (Gothic, Romanesque)
            # Add pointed arches and spires
            img_array = self._add_medieval_patterns(img_array)
        elif class_idx < 15:  # Renaissance styles
            # Add symmetry and classical elements
            img_array = self._add_renaissance_patterns(img_array)
        elif class_idx < 20:  # Modern styles
            # Add clean lines and geometric shapes
            img_array = self._add_modern_patterns(img_array)
        else:  # Contemporary styles
            # Add abstract and experimental elements
            img_array = self._add_contemporary_patterns(img_array)
        
        return img_array
    
    def _add_ancient_patterns(self, img_array: np.ndarray) -> np.ndarray:
        """Add ancient architectural patterns."""
        # Add column-like vertical lines
        for i in range(0, 224, 40):
            img_array[:, i:i+10, :] = [150, 100, 50]  # Brown columns
        
        # Add arch-like curves
        for i in range(50, 174, 60):
            for j in range(50, 174):
                if (j - 112) ** 2 + (i - 87) ** 2 < 1000:
                    img_array[j, i:i+20, :] = [200, 150, 100]  # Light brown arches
        
        return img_array
    
    def _add_medieval_patterns(self, img_array: np.ndarray) -> np.ndarray:
        """Add medieval architectural patterns."""
        # Add pointed arches
        for i in range(50, 174, 60):
            for j in range(50, 174):
                if abs(j - 112) < 30 and (i - 87) ** 2 > 500:
                    img_array[j, i:i+20, :] = [100, 100, 150]  # Blue-gray arches
        
        # Add spires
        for i in range(20, 204, 80):
            img_array[0:50, i:i+10, :] = [80, 80, 120]  # Dark blue spires
        
        return img_array
    
    def _add_renaissance_patterns(self, img_array: np.ndarray) -> np.ndarray:
        """Add renaissance architectural patterns."""
        # Add symmetrical facade
        for i in range(50, 174):
            img_array[i, 50:174, :] = [180, 180, 200]  # Light facade
        
        # Add classical elements
        for i in range(0, 224, 60):
            img_array[100:120, i:i+20, :] = [150, 120, 80]  # Classical frieze
        
        return img_array
    
    def _add_modern_patterns(self, img_array: np.ndarray) -> np.ndarray:
        """Add modern architectural patterns."""
        # Add clean horizontal lines
        for i in range(0, 224, 30):
            img_array[i:i+5, :, :] = [200, 200, 200]  # White lines
        
        # Add geometric shapes
        for i in range(50, 174, 40):
            for j in range(50, 174, 40):
                img_array[j:j+20, i:i+20, :] = [100, 150, 200]  # Blue rectangles
        
        return img_array
    
    def _add_contemporary_patterns(self, img_array: np.ndarray) -> np.ndarray:
        """Add contemporary architectural patterns."""
        # Add abstract patterns
        for i in range(0, 224, 20):
            for j in range(0, 224, 20):
                if random.random() > 0.7:
                    color = np.random.randint(0, 255, 3)
                    img_array[j:j+15, i:i+15, :] = color
        
        # Add curved elements
        for i in range(50, 174):
            for j in range(50, 174):
                if (i - 112) ** 2 + (j - 87) ** 2 < 2000:
                    img_array[j, i, :] = [150, 100, 150]  # Purple curves
        
        return img_array


def create_sample_dataset(data_dir: str = 'data/sample', num_samples: int = 1000):
    """Create a sample dataset for testing."""
    generator = SampleDataGenerator(data_dir)
    generator.generate_sample_dataset(num_classes=25, samples_per_class=num_samples//25)
    return data_dir
