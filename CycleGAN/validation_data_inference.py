import numpy as np
import os
from PIL import Image
from options.test_options import TestOptions
from data import create_dataset
from models import create_model

def save_image(image_tensor, save_path):
    """
    Save a tensor as an image file.
    Args:
        image_tensor (torch.Tensor): The image tensor to save.
        save_path (str): The path to save the image.
    """
    if len(image_tensor.shape) == 4:  # Handle batch tensors
        image_tensor = image_tensor[0]
    if image_tensor.dim() == 3 and image_tensor.shape[0] == 1:  # Grayscale
        image_numpy = image_tensor[0].cpu().numpy()
    elif image_tensor.dim() == 3:  # RGB
        image_numpy = np.transpose(image_tensor.cpu().numpy(), (1, 2, 0))
    else:
        raise ValueError(f"Unexpected tensor shape {image_tensor.shape} for saving as an image.")

    image_numpy = (image_numpy + 1) / 2.0 * 255.0  # Normalize to [0, 255]
    image_numpy = np.clip(image_numpy, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(save_path)

def create_subdirectories(base_dir, image_types):
    """
    Create subdirectories for each image type within the validation folder.
    """
    subdirs = {}
    for image_type in image_types:
        subdir = os.path.join(base_dir, "validation", image_type)
        os.makedirs(subdir, exist_ok=True)
        subdirs[image_type] = subdir
    return subdirs

if __name__ == '__main__':
    opt = TestOptions().parse()

    # Define paths
    opt.dataroot = '/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/Sergio/CycleGAN/datasets/H2L_50k'  # Updated path
    opt.results_dir = '/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/Sergio/CycleGAN/results'
    opt.checkpoints_dir = '/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/Sergio/CycleGAN/checkpoints'
    opt.name = 'H2L_16_50k'
    opt.model = 'cycle_gan'
    opt.no_dropout = True
    opt.epoch = 'latest'
    opt.phase = 'test'  # Use test mode (uses testA and testB)
    opt.num_test = float("inf")  # Process all images


    # Test-specific configurations
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    # Create dataset and model
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    # Create validation directory with subfolders
    base_results_dir = os.path.join(opt.results_dir, opt.name, f"{opt.phase}_{opt.epoch}")
    image_types = ["real_A", "fake_B", "rec_A", "real_B", "fake_A", "rec_B"]
    subdirectories = create_subdirectories(base_results_dir, image_types)

    # Run inference
    if opt.eval:
        model.eval()

    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break

        model.set_input(data)  # Load input data
        model.test()           # Run inference
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()[0]  # Get image filename

        # Save each type of image in its respective directory
        for label, image in visuals.items():
            save_path = os.path.join(subdirectories[label], os.path.basename(img_path))
            save_image(image, save_path)
