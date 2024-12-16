import os
os.sys.path.append(os.getcwd())
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict
from dataset import get_loader
from torch.autograd import Variable
import torchvision.utils as vutils
from model import GEN_DEEP
import cv2

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
    data_name = '/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/Sergio/processed_images/CelebA_data_1000'
    test_loader = get_loader(data_name, opt.batchsize)
    net_G_low2high = GEN_DEEP()
    net_G_low2high = net_G_low2high.cuda()
    a = torch.load('model.pkl')
    net_G_low2high.load_state_dict(a)
    net_G_low2high = net_G_low2high.eval()
    index = 0
    test_file = 'test_res'
    if not os.path.exists(test_file):
        os.makedirs(test_file)
    for idx, data_dict in enumerate(test_loader):
        print(idx)
        index = index + 1
        data_low = data_dict['img16']
        data_high = data_dict['img64']
        img_name = data_dict['imgpath'][0].split('/')[-1]
        with torch.no_grad():
            data_input_low, batchsize_high = to_var(data_low)
            data_input_high, _ = to_var(data_high)
            data_high_output = net_G_low2high(data_input_low)
        np_low = data_low.cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
        np_high = data_high.cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
        np_result = data_high_output.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
        low_path = os.path.join(test_file, f"low_{idx}.jpg")
        high_path = os.path.join(test_file, f"high_{idx}.jpg")
        result_path = os.path.join(test_file, f"result_{idx}.jpg")

        cv2.imwrite(low_path, cv2.cvtColor(np_low, cv2.COLOR_RGB2BGR))
        cv2.imwrite(high_path, cv2.cvtColor(np_high, cv2.COLOR_RGB2BGR))
        cv2.imwrite(result_path, cv2.cvtColor(np_result, cv2.COLOR_RGB2BGR))
        # path = os.path.join(test_file, img_name.split('.')[0]+'.jpg')
        # vutils.save_image(data_high_output.data, path, normalize=True)

if __name__ == '__main__':
    main()
