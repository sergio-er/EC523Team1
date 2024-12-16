import os
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict
from dataset import get_loader
from torch.autograd import Variable
import torchvision.utils as vutils
from model import GEN_DEEP

def to_var(data):
    real_cpu = data
    batchsize = real_cpu.size(0)
    input = Variable(real_cpu.cuda())
    return input, batchsize

def main():
    torch.manual_seed(1)
    np.random.seed(0)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    opt = edict()
    opt.nGPU = 1
    opt.batchsize = 1
    opt.cuda = True
    cudnn.benchmark = True
    print('========================LOAD DATA============================')
    
    # Full path to the dataset folder
    dataset_path = '/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/Sergio/BulatDoubleGan/unpaired_face_sr/Dataset/Dataset/HIGH/wider_lnew'
    test_loader = get_loader(dataset_path, opt.batchsize)
    print(f"Loader created: {len(test_loader)} batches")
    
    # Load the model
    net_G_low2high = GEN_DEEP()
    net_G_low2high = net_G_low2high.cuda()
    model_path = 'model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    a = torch.load(model_path)
    net_G_low2high.load_state_dict(a)
    net_G_low2high = net_G_low2high.eval()
    
    # Define output directories
    base_results_folder = '/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/Sergio/processed_images'
    low_res_folder = os.path.join(base_results_folder, 'low_resolution')
    high_res_folder = os.path.join(base_results_folder, 'high_resolution')
    output_l2h_folder = os.path.join(base_results_folder, 'output_l2h')
    output_h2l_folder = os.path.join(base_results_folder, 'output_h2l')

    # Create directories if they don't exist
    os.makedirs(low_res_folder, exist_ok=True)
    os.makedirs(high_res_folder, exist_ok=True)
    os.makedirs(output_l2h_folder, exist_ok=True)
    os.makedirs(output_h2l_folder, exist_ok=True)
    
    # Process each image in the dataset
    for idx, data_dict in enumerate(test_loader):
        print(f"Processing batch {idx}")
        data_low = data_dict['img16']  # Low-resolution input
        data_high = data_dict['img64']  # High-resolution input
        img_name = data_dict['imgpath'][0].split('/')[-1]
        
        with torch.no_grad():
            # Low-to-High (L2H): Generate high-resolution from low-resolution
            data_input_low, _ = to_var(data_low)
            data_high_output = net_G_low2high(data_input_low)

            # High-to-Low (H2L): Generate low-resolution from high-resolution
            data_input_high, _ = to_var(data_high)
            data_low_output = net_G_low2high(data_input_high)

        # Save images to their respective directories
        vutils.save_image(data_low.data, os.path.join(low_res_folder, img_name.split('.')[0] + '.jpg'), normalize=True)
        print(f"Saved low-resolution image to: {low_res_folder}")

        vutils.save_image(data_high.data, os.path.join(high_res_folder, img_name.split('.')[0] + '.jpg'), normalize=True)
        print(f"Saved high-resolution image to: {high_res_folder}")

        vutils.save_image(data_high_output.data, os.path.join(output_l2h_folder, img_name.split('.')[0] + '.jpg'), normalize=True)
        print(f"Saved L2H output image to: {output_l2h_folder}")

        vutils.save_image(data_low_output.data, os.path.join(output_h2l_folder, img_name.split('.')[0] + '.jpg'), normalize=True)
        print(f"Saved H2L output image to: {output_h2l_folder}")

if __name__ == '__main__':
    main()
