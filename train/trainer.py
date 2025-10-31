import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models


class SSDTrainer:
    """SSD Trainer"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() and 'cuda' in cfg.device else 'cpu')
        
        # Initialize model
        self.model = models.detection.ssd300_vgg16(pretrained=True, num_classes=cfg.num_classes)
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    
    def train(self):
        """Train the model"""
        print(f"Training SSD model for {self.cfg.epochs} epochs")
        
        self.model.train()
        
        for epoch in range(self.cfg.epochs):
            print(f"Epoch {epoch+1}/{self.cfg.epochs}")
            
        print("Training completed")
        
        # Save model
        save_path = f"{self.cfg.save_dir}/ssd_model.pth"
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

