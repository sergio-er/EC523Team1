import os
from glob import glob
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import PIL


class faces_super(data.Dataset):
    def __init__(self, dataset_path, transform):
        # Ensure the dataset path exists
        assert os.path.exists(dataset_path), f"Dataset path does not exist: {dataset_path}"
        self.transform = transform
        self.img_list = []

        # Find all .jpg files in the dataset path
        list_name = glob(os.path.join(dataset_path, "*.jpg"))
        list_name.sort()

        # Raise an error if no images are found
        if not list_name:
            raise FileNotFoundError(f"No .jpg files found in {dataset_path}")

        # Append each image path to the list
        for filename in list_name:
            self.img_list.append(filename)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        data = {}
        # Open the 16x16 image and resize to 64x64
        inp16 = Image.open(self.img_list[index]).convert("RGB")  # Ensure RGB images
        inp64 = inp16.resize((64, 64), resample=PIL.Image.BICUBIC)
        # Apply transformations
        data['img64'] = self.transform(inp64)
        data['img16'] = self.transform(inp16)
        data['imgpath'] = self.img_list[index]
        return data


def get_loader(dataset_path, batch_size=1):
    """
    Returns a DataLoader for the given dataset path.
    
    Args:
    - dataset_path (str): Path to the dataset directory containing .jpg images.
    - batch_size (int): Number of samples per batch. Default is 1.

    Returns:
    - DataLoader: PyTorch DataLoader for the dataset.
    """
    # Define transformations (normalization and conversion to tensor)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Create a dataset instance
    dataset = faces_super(dataset_path, transform)
    
    # Return the DataLoader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,  # Adjust as needed for your system
        pin_memory=True
    )
    return data_loader
