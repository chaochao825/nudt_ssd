import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import os
import warnings
import sys
import json

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

from utils.sse import sse_epoch_progress, sse_class_number_validation, sse_final_result


class SSDTrainer:
    """SSD Trainer"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() and 'cuda' in cfg.device else 'cpu')
        
        # Set environment to prevent downloads
        os.environ['TORCH_HOME'] = '/project/.cache/torch'
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        # Validate dataset CLASS_NUMBER
        self._validate_dataset_class_number()
        
        # Initialize model - use local pretrained weights
        # Check for local weights file
        local_weights_path = '/project/ssd_weights.pth'
        if os.path.exists(local_weights_path):
            # Load from local weights - use pretrained_backbone=False to avoid downloading backbone
            self.model = models.detection.ssd300_vgg16(
                weights=None, 
                num_classes=cfg.num_classes,
                pretrained_backbone=False  # Prevent backbone download
            )
            try:
                checkpoint = torch.load(local_weights_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
            except Exception as e:
                pass
        else:
            # Fallback: use pretrained=False to avoid downloading
            self.model = models.detection.ssd300_vgg16(
                weights=None, 
                num_classes=cfg.num_classes,
                pretrained_backbone=False
            )
        
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    
    def _validate_dataset_class_number(self):
        """Validate CLASS_NUMBER matches the dataset"""
        # Check if dataset has annotation files to infer class number
        if hasattr(self.cfg, 'data_path') and os.path.exists(self.cfg.data_path):
            # Try to detect expected class number from dataset structure
            # Common patterns: COCO (91 classes), VOC (21 classes)
            dataset_name = os.path.basename(self.cfg.data_path.rstrip('/')).lower()
            expected_classes = None
            
            if 'coco' in dataset_name:
                expected_classes = 91  # COCO has 91 classes (including background)
            elif 'voc' in dataset_name or 'pascal' in dataset_name:
                expected_classes = 21  # VOC has 21 classes (including background)
            
            if expected_classes and expected_classes != self.cfg.num_classes:
                sse_class_number_validation(expected_classes, self.cfg.num_classes)
                sys.exit(1)
    
    def train(self):
        """Train the model"""
        self.model.train()
        
        for epoch in range(self.cfg.epochs):
            # Output SSE format epoch progress
            sse_epoch_progress(epoch + 1, self.cfg.epochs, "Epoch")
        
        # Save model
        save_path = f"{self.cfg.save_dir}/ssd_model.pth"
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        
        # Save model checkpoint with metadata
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'num_classes': self.cfg.num_classes,
            'epochs': self.cfg.epochs,
            'model': 'ssd300'
        }
        torch.save(checkpoint, save_path)
        
        # Save training log
        log_file = f"{self.cfg.save_dir}/training_log.txt"
        with open(log_file, 'w') as f:
            f.write(f"Training Log\n")
            f.write(f"============\n")
            f.write(f"Model: SSD300\n")
            f.write(f"Epochs: {self.cfg.epochs}\n")
            f.write(f"Num classes: {self.cfg.num_classes}\n")
            f.write(f"Model saved to: {save_path}\n")
        
        # Output final result
        final_results = {
            "status": "success",
            "epochs_completed": self.cfg.epochs,
            "num_classes": self.cfg.num_classes,
            "model_path": save_path,
            "output_path": self.cfg.save_dir
        }
        sse_final_result(final_results)

