from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms

class PairedDataset(Dataset):
    def __init__(self, hr_dir, lr_dir=None, transform=None, target_size=(64, 64)):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_images = sorted(os.listdir(hr_dir))
        self.lr_images = sorted(os.listdir(lr_dir)) if lr_dir else None
        self.transform = transform or transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizing to [-1, 1]
        ])

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        hr_image = Image.open(hr_path).convert('RGB')

        hr_tensor = self.transform(hr_image)

        if self.lr_dir:
            lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
            lr_image = Image.open(lr_path).convert('RGB')
            lr_tensor = self.transform(lr_image)
            return {'hr': hr_tensor, 'lr': lr_tensor}
        else:
            return {'hr': hr_tensor}

def create_dataset(hr_dir, lr_dir=None, mode='train', target_size=(64, 64)):
    return PairedDataset(hr_dir, lr_dir, target_size=target_size)
