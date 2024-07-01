#calculate PSNR SSIM Learning-rate save
import time
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from math import log10
from skimage import metrics
#from skimage.metrics import structural_similarity as compare_ssim
import cv2

from math import exp
import math
import numpy as np
from torch.autograd import Variable
from torchvision.transforms import ToPILImage

def PSNR_cal(pred_image,gt):
    mse = F.mse_loss(pred_image, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list
def SSIM_cal(pred_image,gt):
    pred_image_list = torch.split(pred_image, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    pred_image_list_np = [pred_image_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    ssim_list = [metrics.structural_similarity(pred_image_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(pred_image_list))]
    return ssim_list

def adjust_learning_rate(optimizer,epoch):
    # --- Decay learning rate --- #
    step = 2 
    if (epoch+1) % step==0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
            print('Learning rate sets to {}.'.format(param_group['lr']))
def save_image_(pred_image,exp_name,image_name):
    pred_image_images = torch.split(pred_image, 1, dim=0)
    batch_num = len(pred_image_images)
    
    for ind in range(batch_num):
        image_name_1 = image_name[ind]      
        save_image(pred_image_images[ind], 'E:/code2023/{}/{}'.format(exp_name, image_name_1))
def store_image(pred_img,exp_name,img_name):
    batch_num = pred_img.shape[0]
    for i in range(batch_num):
        img_input = ((pred_img[i,:,:,:]).cpu().data * 255).numpy()
        img_input = np.clip(img_input, 0, 255)
        img_input = np.transpose(img_input, [1, 2, 0])
        # path='E:/mengxzh/GP/{}/{}'.format(exp_name, img_name[i])
        path = 'G:\SDCnet/{}/{}'.format(exp_name, img_name[i])
        cv2.imwrite(path, img_input)
        


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim_(img1, img2, window_size=11, size_average=True):
    img1=torch.clamp(img1,min=0,max=1)
    img2=torch.clamp(img2,min=0,max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)
def psnr_(pred, gt):
    pred=pred.clamp(0,1).cpu().numpy()
    gt=gt.clamp(0,1).cpu().numpy()
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)

