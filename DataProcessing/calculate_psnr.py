import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

def calculate_psnr_with_scaling(generated_image_path: str, reference_image_path: str) -> float:
    try:
        generated = Image.open(generated_image_path)
        reference = Image.open(reference_image_path)
        
        reference_downsampled = reference.resize((32, 32), Image.BICUBIC)
        reference_upsampled = reference_downsampled.resize((256, 256), Image.BICUBIC)

        fig, ax = plt.subplots(1,3)
        ax[0].imshow(reference)
        ax[1].imshow(reference_upsampled)
        ax[2].imshow(generated)
        
        
        generated_array = np.array(generated)
        reference_array = np.array(reference)
        reference_downsampled_array = np.array(reference_upsampled)
            
        psnr_naive = psnr(reference_array, reference_downsampled_array)
        psnr_real = psnr(reference_array, generated_array)
        return psnr_naive, psnr_real
        
    except Exception as e:
        print(f"Error calculating PSNR: {str(e)}")
        return None


path = '/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/processed_images_CelebA_data_1000'
pathB = 'projectnb/ec523kb/project/teams_Fall_2024/Team_1/processed_images_resshift_out_doubleGAN_h2l'
psnr_naive_all = 0
psnr_real_all = 0
for n in range(1,6):
    psnr_naive, psnr_real = calculate_psnr_with_scaling(path+f'epoch0{n:02d}_fake_B.png', path+f'epoch0{n:02d}_real_A.png')
    psnr_naive_all += psnr_naive/13
    psnr_real_all += psnr_real/13
    print(n, psnr_naive, psnr_real)
print(psnr_naive_all, psnr_real_all)