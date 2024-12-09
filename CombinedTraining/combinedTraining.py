import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cycle_gan_model import CycleGANModel
from CombinedTraining.gaussian_diffusion import GaussianDiffusion
from datasets import create_dataset

# Configuration
EPOCHS_H2L = 50
EPOCHS_L2H = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset paths
DATASET_HR = "/path/to/high_res_dataset"  # 64x64 dataset
DATASET_LR = "/path/to/low_res_dataset"  # 16x16 dataset (only used for validation)
OUTPUT_DIR = "/path/to/output"

# CycleGAN: High-to-Low
def train_cyclegan():
    print("Starting CycleGAN Training: High-to-Low Transformation...")
    opt = {
        'input_nc': 3,  # Number of input channels
        'output_nc': 3,  # Number of output channels
        'ngf': 64,  # Number of generator filters
        'ndf': 64,  # Number of discriminator filters
        'lambda_A': 10.0,
        'lambda_B': 10.0,
        'lambda_identity': 0.5,
        'lr': LEARNING_RATE,
        'beta1': 0.5,
        'device': DEVICE
    }
    model = CycleGANModel(opt)
    train_dataset = create_dataset(DATASET_HR, DATASET_LR, mode='train', target_size=(64, 64))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS_H2L):
        for i, data in enumerate(train_loader):
            model.set_input(data)  # Load data
            model.optimize_parameters()  # Update generators and discriminators
            print(f"Epoch [{epoch}/{EPOCHS_H2L}], Batch [{i}/{len(train_loader)}]")

        # Save model checkpoints
        model.save_networks(f"cyclegan_epoch_{epoch}.pth")

    print("CycleGAN Training Complete!")
    return model

# Gaussian Diffusion: Low-to-High
def train_diffusion(cyclegan_model):
    print("Starting Gaussian Diffusion Training: Low-to-High Super-Resolution...")
    # Generate Low-Res outputs from CycleGAN
    train_dataset = create_dataset(DATASET_HR, mode='train', target_size=(64, 64))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    lr_outputs = []
    for data in train_loader:
        cyclegan_model.set_input(data)
        lr_output = cyclegan_model.fake_B.cpu().detach()
        lr_outputs.append(lr_output)
    lr_dataset = torch.cat(lr_outputs, dim=0)

    # Setup Diffusion Model
    diffusion_model = GaussianDiffusion(
        sqrt_etas=torch.linspace(0.01, 0.99, 1000),
        kappa=1.0,
        model_mean_type="START_X",
        loss_type="MSE",
        sf=4
    ).to(DEVICE)

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS_L2H):
        for i, (lr, hr) in enumerate(zip(lr_dataset, train_dataset)):
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            optimizer.zero_grad()

            # Diffusion Training Loss
            t = torch.randint(0, 1000, (BATCH_SIZE,), device=DEVICE).long()
            loss = diffusion_model.training_losses(
                diffusion_model, hr, lr, t
            )['loss'].mean()

            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch}/{EPOCHS_L2H}], Batch [{i}/{len(lr_dataset)}], Loss: {loss.item()}")

        # Save model checkpoints
        torch.save(diffusion_model.state_dict(), f"{OUTPUT_DIR}/diffusion_epoch_{epoch}.pth")

    print("Diffusion Training Complete!")
    return diffusion_model

if __name__ == "__main__":
    # Train High-to-Low CycleGAN
    cyclegan_model = train_cyclegan()

    # Train Low-to-High Diffusion Model
    diffusion_model = train_diffusion(cyclegan_model)




# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from cycle_gan_model import CycleGANModel
# from gaussian_diffusion import GaussianDiffusion
# from dataset import create_dataset

# # Configuration
# EPOCHS = 50
# BATCH_SIZE = 16
# LEARNING_RATE = 1e-4
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Dataset paths
# DATASET_HR = "/path/to/high_res_dataset"  # 64x64 dataset (HR)
# DATASET_LR = None  # No pre-existing LR dataset needed (generated by CycleGAN)
# OUTPUT_DIR = "/path/to/output"

# # Initialize models
# cyclegan_opt = {
#     'input_nc': 3, 'output_nc': 3, 'ngf': 64, 'ndf': 64, 
#     'lambda_A': 10.0, 'lambda_B': 10.0, 'lambda_identity': 0.5, 
#     'lr': LEARNING_RATE, 'beta1': 0.5, 'device': DEVICE
# }
# cyclegan = CycleGANModel(cyclegan_opt).to(DEVICE)

# diffusion_opt = {
#     'sqrt_etas': torch.linspace(0.01, 0.99, 1000),
#     'kappa': 1.0, 'model_mean_type': "START_X", 
#     'loss_type': "MSE", 'sf': 4
# }
# resshift = GaussianDiffusion(diffusion_opt).to(DEVICE)

# # Optimizers
# optim_cyclegan_G = optim.Adam(cyclegan.get_generator_params(), lr=LEARNING_RATE)
# optim_cyclegan_D = optim.Adam(cyclegan.get_discriminator_params(), lr=LEARNING_RATE)
# optim_resshift = optim.Adam(resshift.parameters(), lr=LEARNING_RATE)

# # Dataset
# train_dataset = create_dataset(DATASET_HR, mode='train', target_size=(64, 64))
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# # Training Loop
# for epoch in range(EPOCHS):
#     for i, data in enumerate(train_loader):
#         hr_images = data['hr'].to(DEVICE)

#         # CycleGAN Forward (H2L)
#         cyclegan.set_input(data)
#         cyclegan.optimize_parameters()
#         cyclegan_loss = cyclegan.get_current_losses()

#         # Extract Low-Res Output from CycleGAN
#         lr_images = cyclegan.fake_B.detach()

#         # ResShift Diffusion Forward (L2H)
#         t = torch.randint(0, 1000, (lr_images.size(0),), device=DEVICE).long()
#         diffusion_loss = resshift.training_losses(resshift, hr_images, lr_images, t)["loss"].mean()

#         # ResShift Optimization
#         optim_resshift.zero_grad()
#         diffusion_loss.backward()
#         optim_resshift.step()

#         # Logging
#         print(f"Epoch {epoch+1}/{EPOCHS}, Batch {i+1}/{len(train_loader)}: "
#               f"CycleGAN Loss = {cyclegan_loss}, ResShift Loss = {diffusion_loss.item():.4f}")

#     # Save Models
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     cyclegan.save_networks(f"{OUTPUT_DIR}/cyclegan_epoch_{epoch+1}.pth")
#     torch.save(resshift.state_dict(), f"{OUTPUT_DIR}/diffusion_epoch_{epoch+1}.pth")

# print("Training complete!")
