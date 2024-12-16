import os
import cv2
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from collections import OrderedDict
import sys
import functools

# Add CycleGAN directory to Python path
CYCLEGAN_PATH = Path(os.path.dirname(os.path.dirname(__file__))) / "CycleGAN"
sys.path.append(str(CYCLEGAN_PATH))

def load_network(network_path, device):
    """Load the trained generator network using the same method as base_model.py
    Args:
        network_path: Path to generator weights
        device: torch device to load model to
    """
    print('loading the model from %s' % network_path)
    
    # Load state dict to specified device
    state_dict = torch.load(network_path, map_location=str(device))
    
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    return state_dict

def process_image(img_path, model, device):
    """Process a single image through the CycleGAN generator
    Args:
        img_path: Path to HR image
        model: Loaded generator model
        device: torch device
    Returns:
        Generated LR image
    """
    # Read and preprocess image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [-1, 1]
    img = (img.astype(np.float32) / 127.5) - 1.0
    
    # Convert to tensor and add batch dimension
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    img = img.to(device)
    
    # Generate LR image
    with torch.no_grad():
        fake_lr = model(img)
    
    # Convert back to numpy and correct range
    fake_lr = fake_lr.cpu().numpy()[0].transpose(1, 2, 0)
    fake_lr = ((fake_lr + 1) * 127.5).clip(0, 255).astype(np.uint8)
    fake_lr = cv2.cvtColor(fake_lr, cv2.COLOR_RGB2BGR)
    
    return fake_lr

def process_dataset(hr_folder, save_dir, model_path):
    """
    Process first 10 images (by filename number) using CycleGAN generator
    Args:
        hr_folder: Path to folder containing HR images
        save_dir: Path to save generated LR images
        model_path: Path to trained CycleGAN generator weights
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the generator model
    from models.networks import ResnetGenerator, define_G
    import torch.nn as nn
    
    # Create model using the same function used in training
    model = define_G(3, 3, 64, 'resnet_9blocks', 'instance', 
                    False, 'normal', 0.02, [0])
    
    # Load state dict using same method as base_model.py
    state_dict = load_network(model_path, device)
    
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
        
    # Patch InstanceNorm checkpoints prior to 0.4
    def patch_instance_norm_state_dict(state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
            
    for key in list(state_dict.keys()):
        patch_instance_norm_state_dict(state_dict, model, key.split('.'))
        
    model.load_state_dict(state_dict)
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    
    hr_folder = Path(hr_folder)
    save_dir = Path(save_dir)
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get and sort image files numerically
    image_files = list(hr_folder.glob('*.png')) + list(hr_folder.glob('*.jpg'))
    
    # Sort files based on the numeric part of their filenames
    def get_number(filepath):
        numbers = ''.join(filter(str.isdigit, filepath.name))
        return float('inf') if not numbers else int(numbers)
    
    image_files.sort(key=get_number)
    
    # Only take first 10 images after sorting
    image_files = image_files[:10]
    
    for idx, img_path in enumerate(image_files):
        print(f"Processing image {idx+1}/10: {img_path.name}")
        
        # Generate LR image using CycleGAN
        lr_img = process_image(img_path, model, device)
        
        # Save LR image
        lr_save_path = save_dir / img_path.name
        cv2.imwrite(str(lr_save_path), lr_img)
        
        print(f"Saved LR image to: {lr_save_path}")

if __name__ == '__main__':
    # Use absolute paths based on project structure
    PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    hr_folder = PROJECT_ROOT / "data/training"
    save_dir = PROJECT_ROOT / "data/training_LR_cyclegan"
    model_path = PROJECT_ROOT / "nina/CycleGAN/checkpoints/H2L_16_50k/90_net_G_B.pth"
    
    print(f"Python path: {sys.path}")
    print(f"Working directory: {os.getcwd()}")
    print(f"CycleGAN path exists: {CYCLEGAN_PATH.exists()}")
    
    process_dataset(hr_folder, save_dir, model_path) 