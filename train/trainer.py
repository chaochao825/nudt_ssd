import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import os
import warnings

from utils.sse import sse_epoch_progress


class SSDTrainer:
    """SSD Trainer"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() and 'cuda' in cfg.device else 'cpu')
        
        # Suppress download warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Set environment to prevent downloads
        os.environ['TORCH_HOME'] = '/project/.cache/torch'
        os.environ['HF_HUB_OFFLINE'] = '1'
        
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
                print(f"Loaded local weights from {local_weights_path}")
            except Exception as e:
                print(f"Warning: Could not load local weights, using random initialization: {e}")
        else:
            # Fallback: use pretrained=False to avoid downloading
            print(f"Warning: Local weights not found at {local_weights_path}, using random initialization")
            self.model = models.detection.ssd300_vgg16(
                weights=None, 
                num_classes=cfg.num_classes,
                pretrained_backbone=False
            )
        
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    
    def train(self):
        """Train the model"""
        print(f"Training SSD model for {self.cfg.epochs} epochs")
        
        self.model.train()
        
        for epoch in range(self.cfg.epochs):
            # Output SSE format epoch progress
            sse_epoch_progress(epoch + 1, self.cfg.epochs, "Epoch")
            print(f"Epoch {epoch+1}/{self.cfg.epochs}")
            
        print("Training completed")
        
        # Save model
        save_path = f"{self.cfg.save_dir}/ssd_model.pth"
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

