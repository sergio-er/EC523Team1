import os, sys
import numpy as np
import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from yoon_data import faces_data, High_Data, Low_Data
from yoon_model import High2Low, Discriminator
from model import GEN_DEEP
from dataset import get_loader


def load_generator(checkpoint_path):
    model = GEN_DEEP().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["G_l2h"])
    model.eval()  # Set model to evaluation mode
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model




if torch.cuda.is_available():
    device = torch.device('cuda')  # Use GPU
    print("Using CUDA")
else:
    device = torch.device('cpu')   # Use CPU
    print("Using CPU")



test_loader = get_loader("/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/nina/Inference/DoubleGAN/LR", batch_size=1)
num_test = 50000
test_save = "/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/nina/Inference/DoubleGAN/HR"

G_l2h = load_generator("intermid_results/models/model_epoch_014.pth")
G_l2h.eval()
for i, sample in enumerate(test_loader):
    if i >= num_test: 
        break
    low_temp = sample["img16"].numpy()
    low = torch.from_numpy(np.ascontiguousarray(low_temp[:, ::-1, :, :])).to(device)
    with torch.no_grad():
        hign_gen = G_l2h(low)
    np_low = low.cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
    np_gen = hign_gen.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
    np_low = (np_low - np_low.min()) / (np_low.max() - np_low.min())
    np_gen = (np_gen - np_gen.min()) / (np_gen.max() - np_gen.min())
    np_low = (np_low * 255).astype(np.uint8)
    np_gen = (np_gen * 255).astype(np.uint8)
    # cv2.imwrite("{}/imgs/{}_{}_lr.png".format(test_save, i+1), np_low)
    cv2.imwrite("{}/{}_sr.png".format(test_save, i+1), np_gen)

    output_filename = os.path.basename(sample["imgpath"][0])
    output_path = os.path.join(test_save, output_filename)
    cv2.imwrite(output_path, np_gen)

print("saved files ")