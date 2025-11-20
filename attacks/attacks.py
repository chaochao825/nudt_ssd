import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import glob
import warnings
import sys
import json

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

from utils.sse import sse_adv_samples_gen_validated, sse_class_number_validation, sse_final_result


class SSDDataset(Dataset):
    """Simple dataset for loading images"""
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend(glob.glob(os.path.join(data_path, '**', ext), recursive=True))
        
        if not self.image_files:
            raise ValueError(f"No images found in {data_path}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {'img': image, 'path': img_path, 'idx': idx}


class SSDAttacks:
    """SSD Attack implementation"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() and 'cuda' in cfg.device else 'cpu')
        
        # Suppress warnings to avoid model download warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', message='.*Downloading.*')
        
        # Set environment to prevent downloads
        os.environ['TORCH_HOME'] = '/project/.cache/torch'
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        # Validate CLASS_NUMBER before loading model
        self._validate_class_number()
        
        # Load pretrained SSD model (using torchvision's implementation)
        if hasattr(cfg, 'pretrained') and os.path.exists(cfg.pretrained):
            # Load custom weights
            self.model = self._load_ssd_model(cfg.pretrained)
        else:
            # Try to use local weights first
            local_weights_path = '/project/ssd_weights.pth'
            if os.path.exists(local_weights_path):
                self.model = self._load_ssd_model(local_weights_path)
            else:
                # Use weights=None to avoid downloading
                self.model = models.detection.ssd300_vgg16(
                    weights=None, 
                    num_classes=cfg.num_classes,
                    pretrained_backbone=False
                )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Data transforms
        self.transform = T.Compose([
            T.Resize((300, 300)),
            T.ToTensor(),
        ])
    
    def _validate_class_number(self):
        """Validate CLASS_NUMBER matches the model/dataset"""
        if hasattr(self.cfg, 'pretrained') and os.path.exists(self.cfg.pretrained):
            try:
                checkpoint = torch.load(self.cfg.pretrained, map_location='cpu')
                # Try to extract class number from checkpoint
                if isinstance(checkpoint, dict):
                    if 'num_classes' in checkpoint:
                        expected_classes = checkpoint['num_classes']
                        if expected_classes != self.cfg.num_classes:
                            sse_class_number_validation(expected_classes, self.cfg.num_classes)
                            sys.exit(1)
                    elif 'state_dict' in checkpoint:
                        # Try to infer from model architecture
                        state_dict = checkpoint['state_dict']
                        self._check_class_number_from_state_dict(state_dict)
                else:
                    # state_dict directly
                    self._check_class_number_from_state_dict(checkpoint)
            except Exception as e:
                # If we can't validate, just continue
                pass
    
    def _check_class_number_from_state_dict(self, state_dict):
        """Check class number from state dict"""
        # Look for classification head parameters
        for key in state_dict.keys():
            if 'num_classes' in key or 'classifier' in key:
                # Try to infer number of classes from classifier layer
                if 'weight' in key and len(state_dict[key].shape) >= 2:
                    num_classes_in_model = state_dict[key].shape[0]
                    # COCO has 91 classes, Pascal VOC has 21
                    if num_classes_in_model in [91, 21, 80, 20]:
                        if num_classes_in_model != self.cfg.num_classes:
                            sse_class_number_validation(num_classes_in_model, self.cfg.num_classes)
                            sys.exit(1)
                    break
    
    def _load_ssd_model(self, model_path):
        """Load SSD model from checkpoint"""
        model = models.detection.ssd300_vgg16(
            weights=None, 
            num_classes=self.cfg.num_classes,
            pretrained_backbone=False
        )
        if model_path.endswith('.pth') or model_path.endswith('.pt'):
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        return model
    
    def get_dataloader(self):
        """Create dataloader for images"""
        dataset = SSDDataset(self.cfg.data_path, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.cfg.batch, 
                              shuffle=False, num_workers=self.cfg.workers)
        return dataloader
    
    def preprocess(self, batch):
        """Preprocess batch"""
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        return batch
    
    def run_adv(self, args):
        """Generate adversarial samples"""
        os.makedirs(f'{self.cfg.save_dir}/adv_images', exist_ok=True)
        dataloader = self.get_dataloader()
        
        total_images = 0
        for batch_i, batch in enumerate(dataloader):
            batch = self.preprocess(batch)
            
            if args.attack_method == 'pgd':
                adv_images = self.pgd(batch['img'], eps=args.epsilon, alpha=args.step_size, 
                                     steps=args.max_iterations, random_start=args.random_start)
            elif args.attack_method == 'fgsm':
                adv_images = self.fgsm(batch['img'], eps=args.epsilon)
            elif args.attack_method == 'cw':
                adv_images = self.cw(batch['img'], c=1, kappa=0, steps=args.max_iterations, lr=0.01)
            elif args.attack_method == 'bim':
                adv_images = self.bim(batch['img'], eps=args.epsilon, alpha=args.step_size, 
                                     steps=args.max_iterations)
            elif args.attack_method == 'deepfool':
                adv_images, _ = self.deepfool(batch['img'], steps=args.max_iterations, overshoot=0.02)
            else:
                raise ValueError(f'Invalid attack method: {args.attack_method}')
            
            # Save adversarial images
            to_pil = transforms.ToPILImage()
            pil_image = to_pil(adv_images[0].cpu())
            adv_image_name = f'{self.cfg.save_dir}/adv_images/adv_image_{batch_i}.jpg'
            pil_image.save(adv_image_name)
            sse_adv_samples_gen_validated(adv_image_name)
            total_images += 1
        
        # Output final result
        final_results = {
            "status": "success",
            "total_adversarial_samples": total_images,
            "attack_method": args.attack_method,
            "epsilon": float(args.epsilon),
            "output_path": f'{self.cfg.save_dir}/adv_images'
        }
        sse_final_result(final_results)
    
    def run_attack(self, args):
        """Evaluate model robustness"""
        # Create output directories
        os.makedirs(f'{self.cfg.save_dir}/attack_results', exist_ok=True)
        os.makedirs(f'{self.cfg.save_dir}/adversarial_images', exist_ok=True)
        
        dataloader = self.get_dataloader()
        total = 0
        successful_attacks = 0
        total_clean_detections = 0
        total_adv_detections = 0
        clean_confidence_sum = 0.0
        adv_confidence_sum = 0.0
        
        results_log = []
        
        for batch_i, batch in enumerate(dataloader):
            batch = self.preprocess(batch)
            
            # Get clean predictions
            with torch.no_grad():
                clean_outputs = self.model(batch['img'])
            
            # Generate adversarial examples
            if args.attack_method == 'fgsm':
                adv_images = self.fgsm(batch['img'], eps=args.epsilon)
            else:
                adv_images = self.pgd(batch['img'], eps=args.epsilon, alpha=args.step_size,
                                     steps=args.max_iterations, random_start=args.random_start)
            
            # Save adversarial images
            for i in range(adv_images.shape[0]):
                to_pil = transforms.ToPILImage()
                pil_image = to_pil(adv_images[i].cpu())
                adv_image_path = f'{self.cfg.save_dir}/adversarial_images/adv_{batch_i}_{i}.jpg'
                pil_image.save(adv_image_path)
            
            # Get adversarial predictions
            with torch.no_grad():
                adv_outputs = self.model(adv_images)
            
            # Calculate metrics
            for clean_out, adv_out in zip(clean_outputs, adv_outputs):
                num_clean_det = len(clean_out['boxes'])
                num_adv_det = len(adv_out['boxes'])
                
                total_clean_detections += num_clean_det
                total_adv_detections += num_adv_det
                
                if num_clean_det > 0:
                    clean_confidence_sum += clean_out['scores'].mean().item()
                if num_adv_det > 0:
                    adv_confidence_sum += adv_out['scores'].mean().item()
                
                # Count as successful attack if detections dropped
                if num_adv_det < num_clean_det:
                    successful_attacks += 1
                
                results_log.append({
                    'image_idx': total,
                    'clean_detections': num_clean_det,
                    'adv_detections': num_adv_det,
                    'attack_success': num_adv_det < num_clean_det
                })
            
            total += batch['img'].size(0)
        
        # Calculate final metrics
        attack_success_rate = successful_attacks / total if total > 0 else 0
        avg_clean_confidence = clean_confidence_sum / total if total > 0 else 0
        avg_adv_confidence = adv_confidence_sum / total if total > 0 else 0
        detection_drop_rate = (total_clean_detections - total_adv_detections) / total_clean_detections if total_clean_detections > 0 else 0
        
        # Save detailed results
        results_file = f'{self.cfg.save_dir}/attack_results/evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results_log, f, indent=2)
        
        # Save summary
        summary_file = f'{self.cfg.save_dir}/attack_results/summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Attack Evaluation Summary\n")
            f.write(f"========================\n")
            f.write(f"Total samples: {total}\n")
            f.write(f"Successful attacks: {successful_attacks}\n")
            f.write(f"Attack success rate: {attack_success_rate:.2%}\n")
            f.write(f"Detection drop rate: {detection_drop_rate:.2%}\n")
            f.write(f"Avg clean confidence: {avg_clean_confidence:.4f}\n")
            f.write(f"Avg adv confidence: {avg_adv_confidence:.4f}\n")
        
        # Output final result as SSE
        final_results = {
            "status": "success",
            "total_samples": total,
            "successful_attacks": successful_attacks,
            "attack_success_rate": round(attack_success_rate, 4),
            "detection_drop_rate": round(detection_drop_rate, 4),
            "avg_clean_confidence": round(avg_clean_confidence, 4),
            "avg_adv_confidence": round(avg_adv_confidence, 4),
            "attack_method": args.attack_method,
            "epsilon": float(args.epsilon),
            "output_path": self.cfg.save_dir
        }
        sse_final_result(final_results)
    
    def fgsm(self, images, eps=8/255):
        """FGSM attack"""
        images.requires_grad = True
        
        # Forward pass
        outputs = self.model(images)
        
        # Calculate loss (use detection loss)
        loss = self._calculate_detection_loss(outputs, images)
        
        # Backward pass
        loss.backward()
        
        # Generate adversarial examples
        grad = images.grad.data
        adv_images = images + eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        
        return adv_images
    
    def pgd(self, images, eps=8/255, alpha=2/255, steps=10, random_start=True):
        """PGD attack"""
        adv_images = images.clone().detach()
        
        if random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        
        for _ in range(steps):
            adv_images.requires_grad = True
            
            outputs = self.model(adv_images)
            loss = self._calculate_detection_loss(outputs, adv_images)
            
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
            
            adv_images = adv_images.detach() + alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        
        return adv_images
    
    def bim(self, images, eps=8/255, alpha=2/255, steps=10):
        """BIM attack"""
        if steps == 0:
            steps = int(min(eps * 255 + 4, 1.25 * eps * 255))
        
        ori_images = images.clone().detach()
        
        for _ in range(steps):
            images.requires_grad = True
            
            outputs = self.model(images)
            loss = self._calculate_detection_loss(outputs, images)
            
            grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
            
            adv_images = images + alpha * grad.sign()
            a = torch.clamp(ori_images - eps, min=0)
            b = (adv_images >= a).float() * adv_images + (adv_images < a).float() * a
            c = (b > ori_images + eps).float() * (ori_images + eps) + (b <= ori_images + eps).float() * b
            images = torch.clamp(c, max=1).detach()
        
        return images
    
    def cw(self, images, c=1, kappa=0, steps=50, lr=0.01):
        """C&W attack"""
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True
        
        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        
        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()
        
        optimizer = optim.Adam([w], lr=lr)
        
        for step in range(steps):
            adv_images = self.tanh_space(w)
            
            current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()
            
            outputs = self.model(adv_images)
            f_loss = self._calculate_detection_loss(outputs, adv_images)
            
            cost = L2_loss + c * f_loss
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            mask = (current_L2.detach() < best_L2).float()
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2
            
            mask = mask.view([-1] + [1] * (len(images.shape) - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images
        
        return best_adv_images
    
    def deepfool(self, images, steps=50, overshoot=0.02):
        """DeepFool attack"""
        batch_size = len(images)
        adv_images = []
        
        for idx in range(batch_size):
            image = images[idx:idx+1].clone().detach()
            
            for _ in range(steps):
                image.requires_grad = True
                
                outputs = self.model(image)
                loss = self._calculate_detection_loss(outputs, image)
                
                grad = torch.autograd.grad(loss, image, retain_graph=False, create_graph=False)[0]
                
                # Apply perturbation
                pert = (1 + overshoot) * grad.sign() * 0.02
                image = image.detach() + pert
                image = torch.clamp(image, min=0, max=1)
            
            adv_images.append(image)
        
        return torch.cat(adv_images), None
    
    def _calculate_detection_loss(self, outputs, images):
        """Calculate detection loss for adversarial training"""
        # For SSD, outputs is a list of dicts with 'boxes', 'labels', 'scores'
        # We use a simple loss based on confidence scores
        total_loss = 0
        for output in outputs:
            if 'scores' in output and len(output['scores']) > 0:
                # Maximize confidence (for untargeted attack)
                total_loss += output['scores'].sum()
        
        if total_loss == 0:
            # If no detections, use a dummy loss
            total_loss = images.sum() * 0
        
        return total_loss
    
    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)
    
    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))
    
    def inverse_tanh_space(self, x):
        return self.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

