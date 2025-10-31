import torch
import torch.nn as nn
import os
from PIL import Image, ImageFilter
import glob
from torchvision import transforms
import numpy as np
from utils.sse import sse_clean_samples_gen_validated


class SSDDefends:
    """SSD Defense mechanisms"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cpu')  # Defense typically on CPU
    
    def run_defend(self, args):
        """Apply defense mechanisms to images"""
        os.makedirs(f'{self.cfg.save_dir}/defended_images', exist_ok=True)
        
        # Find all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.cfg.data_path, '**', ext), recursive=True))
        
        for img_path in image_files:
            image = Image.open(img_path).convert('RGB')
            
            if args.defend_method == 'scale':
                defended_image = self.scale_defense(image)
            elif args.defend_method == 'comp':
                defended_image = self.compression_defense(image)
            elif args.defend_method == 'neural_cleanse':
                defended_image = self.neural_cleanse_defense(image)
            elif args.defend_method == 'pgd':
                defended_image = self.pgd_defense(image)
            elif args.defend_method == 'fgsm':
                defended_image = self.fgsm_defense(image)
            else:
                defended_image = image
            
            # Save defended image
            filename = os.path.basename(img_path)
            save_path = f'{self.cfg.save_dir}/defended_images/{filename}'
            defended_image.save(save_path)
            sse_clean_samples_gen_validated(save_path)
    
    def scale_defense(self, image):
        """Scale-based defense"""
        # Resize down and up to remove adversarial perturbations
        original_size = image.size
        scale_factor = 0.5
        
        new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
        image = image.resize(new_size, Image.BILINEAR)
        image = image.resize(original_size, Image.BILINEAR)
        
        return image
    
    def compression_defense(self, image):
        """JPEG compression defense"""
        # Apply JPEG compression to remove perturbations
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=75)
        buffer.seek(0)
        image = Image.open(buffer)
        
        return image
    
    def neural_cleanse_defense(self, image):
        """Neural Cleanse defense - removes potential triggers"""
        # Apply Gaussian blur to remove small perturbations
        original_size = image.size
        
        # Apply blur
        blurred = image.filter(ImageFilter.GaussianBlur(radius=1.5))
        
        # Slight sharpening to preserve details
        sharpened = blurred.filter(ImageFilter.SHARPEN)
        
        # Combine original and processed (weighted average)
        img_array = np.array(image, dtype=np.float32)
        sharp_array = np.array(sharpened, dtype=np.float32)
        
        # 80% sharpened + 20% original
        combined = (sharp_array * 0.8 + img_array * 0.2).astype(np.uint8)
        
        return Image.fromarray(combined)
    
    def pgd_defense(self, image):
        """PGD-based adversarial training defense"""
        # Apply multiple small perturbations and average
        # This simulates adversarial training effect
        img_array = np.array(image, dtype=np.float32)
        
        # Add small random noise (simulating PGD)
        noise_scale = 8  # Similar to epsilon in attacks
        noise = np.random.uniform(-noise_scale, noise_scale, img_array.shape)
        
        # Apply noise and clip
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Apply median filter to smooth
        result_img = Image.fromarray(noisy)
        result_img = result_img.filter(ImageFilter.MedianFilter(size=3))
        
        return result_img
    
    def fgsm_defense(self, image):
        """FGSM-based defense - gradient masking"""
        # Apply bit depth reduction and quantization
        img_array = np.array(image, dtype=np.float32)
        
        # Reduce bit depth (8-bit to 6-bit equivalent)
        quantized = np.round(img_array / 4) * 4
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)
        
        # Apply slight blur to mask gradients
        result_img = Image.fromarray(quantized)
        result_img = result_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return result_img

