import os
import cv2
import torch
import random
import numpy as np
from pathlib import Path
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
import torch.nn.functional as F
import math
from scipy import special

def generate_kernel1d(kernel_size, sigma, rate_iso=1.0):
    """Generate 1D Gaussian or isotropic kernel
    Args:
        kernel_size (int):
        sigma (float):
        rate_iso (float): rate of isotropic kernel
    Returns:
        kernel_1d (torch.Tensor): (kernel_size, )
    """
    center = kernel_size // 2
    kernel_1d = torch.zeros(kernel_size)
    for i in range(kernel_size):
        kernel_1d[i] = ((i - center)**2) / (2 * sigma * sigma)
    kernel_1d = (-kernel_1d).exp()
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    return kernel_1d

def random_bivariate_gaussian_kernel(kernel_size, sigma_x, sigma_y, rotation):
    """Generate random bivariate isotropic or anisotropic Gaussian kernel
    Args:
        kernel_size (int):
        sigma_x (float):
        sigma_y (float):
        rotation (float): Rotation in degrees
    Returns:
        kernel (torch.Tensor): (kernel_size, kernel_size)
    """
    center = kernel_size // 2
    x = torch.arange(-center, center + 1).float()
    y = torch.arange(-center, center + 1).float()
    y, x = torch.meshgrid(y, x)
    
    # Rotation
    rotation = rotation * math.pi / 180
    cos_theta = math.cos(rotation)
    sin_theta = math.sin(rotation)
    
    x_rot = cos_theta * x - sin_theta * y
    y_rot = sin_theta * x + cos_theta * y
    
    kernel = torch.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2)))
    kernel = kernel / kernel.sum()
    
    return kernel

def generate_kernel(kernel_size, sigma_x, sigma_y=None, rotation=0):
    """Generate kernel based on config settings"""
    if sigma_y is None:
        sigma_y = sigma_x
    
    kernel = random_bivariate_gaussian_kernel(kernel_size, sigma_x, sigma_y, rotation)
    return kernel.unsqueeze(0).unsqueeze(0)

def generate_sinc_kernel(kernel_size, omega_c):
    """Generate sinc kernel
    Args:
        kernel_size (int):
        omega_c (float): Cutoff frequency in radians
    Returns:
        kernel (torch.Tensor): (kernel_size, kernel_size)
    """
    center = kernel_size // 2
    x = torch.arange(-center, center + 1).float()
    y = torch.arange(-center, center + 1).float()
    y, x = torch.meshgrid(y, x)
    
    r = torch.sqrt(x**2 + y**2)
    kernel = omega_c * special.j1(omega_c * r) / (2 * math.pi * r)
    kernel[center, center] = omega_c**2 / (4 * math.pi)
    
    return kernel.unsqueeze(0).unsqueeze(0)

class DegradationPipeline:
    def __init__(self, configs):
        self.configs = configs
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.use_sharpener = USMSharp().cuda()
        
    def get_random_kernel(self):
        """Get random kernel based on kernel list"""
        kernel_size = self.configs.blur_kernel_size
        kernel_type = random.choices(self.configs.kernel_list, self.configs.kernel_prob)[0]
        
        sigma = random.uniform(*self.configs.blur_sigma)
        
        if kernel_type == 'iso':
            kernel = generate_kernel(kernel_size, sigma)
        elif kernel_type == 'aniso':
            sigma_x = random.uniform(*self.configs.blur_sigma)
            sigma_y = random.uniform(*self.configs.blur_sigma)
            rotation = random.uniform(0, 360)
            kernel = generate_kernel(kernel_size, sigma_x, sigma_y, rotation)
        elif kernel_type in ['generalized_iso', 'generalized_aniso']:
            # Implementation of generalized Gaussian kernel if needed
            kernel = generate_kernel(kernel_size, sigma)
        elif kernel_type in ['plateau_iso', 'plateau_aniso']:
            # Implementation of plateau kernel if needed
            kernel = generate_kernel(kernel_size, sigma)
            
        return kernel.cuda()
        
    @torch.no_grad()
    def __call__(self, im_gt):
        """
        Args:
            im_gt: RGB image in range [0, 1], shape: (H, W, C)
        Returns:
            im_lq: Degraded LR image
        """
        # Convert to tensor and move to GPU
        im_gt = torch.from_numpy(im_gt.transpose(2, 0, 1)).unsqueeze(0).float().cuda()
        
        ori_h, ori_w = im_gt.size()[2:4]
        print(f"Original image size: {ori_h}x{ori_w}")
        
        # Fixed scale factor of 4
        sf = self.configs.sf
        
        if self.configs.use_sharp:
            im_gt = self.use_sharpener(im_gt)

        # First degradation
        kernel1 = self.get_random_kernel()
        out = filter2D(im_gt, kernel1)
        
        # Random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.configs.resize_prob)[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.configs.resize_range[1])
        elif updown_type == 'down':
            scale = random.uniform(self.configs.resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        # Add noise
        if random.random() < self.configs.gaussian_noise_prob:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.configs.noise_range,
                clip=True,
                rounds=False,
                gray_prob=self.configs.gray_noise_prob
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.configs.poisson_scale_range,
                gray_prob=self.configs.gray_noise_prob,
                clip=True,
                rounds=False
            )

        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.jpeg_range)
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p)

        # Second degradation
        if random.random() < self.configs.second_order_prob:
            # Get second kernel with different settings
            kernel2 = self.get_random_kernel()  # This will use blur_kernel_size2, etc.
            
            if random.random() < self.configs.second_blur_prob:
                out = filter2D(out, kernel2)
            
            updown_type = random.choices(['up', 'down', 'keep'], self.configs.resize_prob2)[0]
            if updown_type == 'up':
                scale = random.uniform(1, self.configs.resize_range2[1])
            elif updown_type == 'down':
                scale = random.uniform(self.configs.resize_range2[0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out,
                size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
                mode=mode
            )

            # Add noise
            if random.random() < self.configs.gaussian_noise_prob2:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.configs.noise_range2,
                    clip=True,
                    rounds=False,
                    gray_prob=self.configs.gray_noise_prob2
                )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.configs.poisson_scale_range2,
                    gray_prob=self.configs.gray_noise_prob2,
                    clip=True,
                    rounds=False
                )

        # Generate sinc kernel
        if random.random() < self.configs.final_sinc_prob:
            omega_c = random.uniform(np.pi / 3, np.pi)
            sinc_kernel = generate_sinc_kernel(self.configs.blur_kernel_size, omega_c).cuda()
        else:
            sinc_kernel = generate_kernel(self.configs.blur_kernel_size, 
                                        random.uniform(*self.configs.blur_sigma)).cuda()

        # Final steps (random order of JPEG + sinc filter)
        if random.random() < 0.5:
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // sf, ori_w // sf), mode=mode)
            out = filter2D(out, sinc_kernel)
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // sf, ori_w // sf), mode=mode)
            out = filter2D(out, sinc_kernel)

        # Final resize if needed
        if self.configs.resize_back:
            out = F.interpolate(out, size=(ori_h, ori_w), mode='bicubic')

        # Convert back to numpy array
        im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        im_lq = im_lq[0].cpu().numpy().transpose(1, 2, 0)
        print(f"Final LR image size: {im_lq.shape[0]}x{im_lq.shape[1]}")
        
        return im_lq

def process_dataset(hr_folder, save_dir, configs):
    """
    Process first 10 images (by filename number) in the HR folder and create corresponding LR pairs
    Args:
        hr_folder: Path to folder containing HR images
        save_dir: Path to save degraded LR images
        configs: Configuration object
    """
    degrader = DegradationPipeline(configs)
    hr_folder = Path(hr_folder)
    save_dir = Path(save_dir)
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get and sort image files numerically
    image_files = list(hr_folder.glob('*.png')) + list(hr_folder.glob('*.jpg'))
    
    # Sort files based on the numeric part of their filenames
    def get_number(filepath):
        # Extract numbers from filename, default to infinity if no number found
        numbers = ''.join(filter(str.isdigit, filepath.name))
        return float('inf') if not numbers else int(numbers)
    
    image_files.sort(key=get_number)
    
    # Only take first 10 images after sorting
    image_files = image_files[:10]
    
    for idx, img_path in enumerate(image_files):
        print(f"Processing image {idx+1}/10: {img_path.name}")
        
        # Read HR image
        hr_img = cv2.imread(str(img_path))
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB) / 255.0
        
        # Generate LR image
        lr_img = degrader(hr_img)
        
        # Save LR image with same name as HR
        lr_save_path = save_dir / img_path.name
        cv2.imwrite(str(lr_save_path), cv2.cvtColor((lr_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    # Configuration matching realsr_swinunet_realesrgan256_journal.yaml
    configs = type('Config', (), {
        # Degradation settings from yaml
        'sf': 4,
        'use_sharp': False,  # matches yaml's use_sharp: False
        
        # First degradation process
        'resize_prob': [0.2, 0.7, 0.1],
        'resize_range': [0.15, 1.5],
        'gaussian_noise_prob': 0.5,
        'noise_range': [1, 30],
        'poisson_scale_range': [0.05, 3.0],
        'gray_noise_prob': 0.4,
        'jpeg_range': [30, 95],
        
        # Second degradation process
        'second_order_prob': 0.5,  # matches yaml's second_order_prob: 0.5
        'second_blur_prob': 0.8,
        'resize_prob2': [0.3, 0.4, 0.3],
        'resize_range2': [0.3, 1.2],
        'gaussian_noise_prob2': 0.5,
        'noise_range2': [1, 25],
        'poisson_scale_range2': [0.05, 2.5],
        'gray_noise_prob2': 0.4,
        'jpeg_range2': [30, 95],
        
        # Other settings
        'resize_back': False,  # matches yaml's resize_back: False
        
        # Kernel settings from yaml's data.train.params
        'blur_kernel_size': 15,
        'kernel_list': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        'kernel_prob': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        'sinc_prob': 0.1,
        'blur_sigma': [0.2, 3.0],
        'betag_range': [0.5, 4.0],
        'betap_range': [1, 2.0],
        
        # Second kernel settings
        'blur_kernel_size2': 15,
        'kernel_list2': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        'kernel_prob2': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        'sinc_prob2': 0.1,
        'blur_sigma2': [0.2, 1.5],
        'betag_range2': [0.5, 4.0],
        'betap_range2': [1, 2.0],
        
        'final_sinc_prob': 0.8,
    })

    # Usage
    hr_folder = "data/training"
    save_dir = "data/training_LR_realesrgan"
    process_dataset(hr_folder, save_dir, configs) 