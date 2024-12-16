import os
import numpy as np
import torch
import cv2
from yoon_model import High2Low  # Import the generator model class
from glob import glob
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



# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load model weights
def load_generator(checkpoint_path):
    model = High2Low().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["G_h2l"])
    model.eval()  # Set model to evaluation mode
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model

# Function to generate low-resolution images
def generate_low_resolution(model, input_image_path, output_dir):
    print("start generating low res.")
    os.makedirs(output_dir, exist_ok=True)
    assert os.access(output_dir, os.W_OK), f"Cannot write to directory: {output_dir}"
    test_loader = get_loader(input_image_path, batch_size=1)
    print(f"Finished loading data. Total samples: {len(test_loader.dataset)}\n")

    num_test = 50000
    z = torch.randn(1, 64, dtype=torch.float32).to(device)
    for i, sample in enumerate(test_loader):
        if i >= num_test: 
                break
        high_temp = sample["img64"].numpy()
        # print(high_temp.shape)
        high = torch.from_numpy(np.ascontiguousarray(high_temp[:, ::-1, :, :])).to(device)
        with torch.no_grad():
            low_gen = model(high,z)
        np_high = high.cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
        np_gen = low_gen.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
        np_high = (np_high - np_high.min()) / (np_high.max() - np_high.min())
        np_gen = (np_gen - np_gen.min()) / (np_gen.max() - np_gen.min())
        np_high = (np_high * 255).astype(np.uint8)
        np_gen = (np_gen * 255).astype(np.uint8)


        # print(f"np_high shape: {np_high.shape}, dtype: {np_high.dtype}")
        # print(f"np_gen shape: {np_gen.shape}, dtype: {np_gen.dtype}")

        # path_lr = "{}/{}_lr.png".format(output_dir, i+1)
        # path_sr = "{}/{}_sr.png".format(output_dir, i+1)
        # print(f"Saving low-resolution image to: {path_lr}")
        # print(f"Saving super-resolution image to: {path_sr}")

        # cv2.imwrite("{}/{}_lr.jpg".format(output_dir, i+1), np_high)
        # cv2.imwrite("{}/{}_lr.jpg".format(output_dir, i+1), np_gen)
        # print("image saved")
        output_filename = os.path.basename(sample["imgpath"][0])
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, np_gen)

if __name__ == "__main__":
    # Path to the checkpoint file
    checkpoint_path = "intermid_results/models/model_epoch_014.pth"
    
    # Path to input high-resolution image
    input_image_path = "/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/nina/Inference/Originals/CelebA_50k"
    
    # Output directory to save the low-resolution image
    output_dir = "/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/nina/Inference/DoubleGAN/LR"
    
    # Load the generator and generate low-resolution image
    generator = load_generator(checkpoint_path)
    generate_low_resolution(generator, input_image_path, output_dir)
    
