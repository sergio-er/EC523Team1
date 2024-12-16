import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

def calculate_psnr_with_scaling(generated_image_path: str, reference_image_path: str):
    """
    Calculate PSNR values between the generated image and the reference image, 
    including a downsampled/upscaled comparison.
    """
    try:
        # Load the images
        generated = Image.open(generated_image_path)
        reference = Image.open(reference_image_path)
        
        # Resize both images to the same size (e.g., 256x256)
        target_size = (16, 16)  # You can change this size if needed
        # generated_resized = generated.resize(target_size, Image.BICUBIC)
        reference_resized = reference.resize(target_size, Image.BICUBIC)
        
        # Downsample and upsample the reference image
        reference_downsampled = reference_resized.resize((16, 16), Image.BICUBIC)
        # reference_upsampled = reference_downsampled.resize((64, 64), Image.BICUBIC)

        # # Display the images (optional, useful for debugging)
        # fig, ax = plt.subplots(1, 3, figsize=(10, 4))
        # ax[0].imshow(reference_resized)
        # ax[0].set_title("Reference Image")
        # ax[1].imshow(reference_upsampled)
        # ax[1].set_title("Upsampled Image")
        # ax[2].imshow(generated_resized)
        # ax[2].set_title("Generated Image")
        # plt.show()
        
        # Convert images to numpy arrays
        generated_array = np.array(generated)
        reference_array = np.array(reference_resized)
        reference_downsampled_array = np.array(reference_downsampled)
        
        # Calculate PSNR values
        # psnr_naive = psnr(reference_array, reference_upsampled_array)
        psnr_real = psnr(reference_downsampled_array, generated_array)
        
        return psnr_real
    
    except Exception as e:
        print(f"Error calculating PSNR for {generated_image_path} and {reference_image_path}: {str(e)}")
        return None, None

# Paths to the directories containing the images
pathA = '/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/nina/Inference/Originals/CelebA_50k'  # Reference 
pathB = '/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/nina/Inference/CycleGAN/LR'  # Generated 

# List the files in each directory, filter for images, and sort them
filesA = sorted([f for f in os.listdir(pathA) if f.endswith(('.jpg', '.png'))])
filesB = sorted([f for f in os.listdir(pathB) if f.endswith(('.jpg', '.png'))])

# Find common files in both directories based on file names
common_files = set(filesA).intersection(filesB)

# Initialize accumulators for average PSNR
psnr_naive_all = 0
psnr_real_all = 0
count = 0

# Iterate through the common files
for filename in common_files:
    fileA_path = os.path.join(pathA, filename)
    fileB_path = os.path.join(pathB, filename)
    
    psnr_real = calculate_psnr_with_scaling(fileB_path, fileA_path)
    if  psnr_real is not None:
        # psnr_naive_all += psnr_naive
        psnr_real_all += psnr_real
        count += 1
        print(f"{count}: {filename} - Real PSNR = {psnr_real:.2f}")

# Compute the average PSNR
if count > 0:
    # psnr_naive_avg = psnr_naive_all / count
    psnr_real_avg = psnr_real_all / count
    # print(f"Average Naive PSNR: {psnr_naive_avg:.2f}")
    print(f"Average Real PSNR: {psnr_real_avg:.2f}")
else:
    print("No matching files found for PSNR calculation.")
