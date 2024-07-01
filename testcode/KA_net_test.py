import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from view_testdata1 import TestData#from test_data_real import TestData
from transweather_model import Transweather
from LD_model1 import Dehaze
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import os
import time
import numpy as np
from Utils import save_image_,PSNR_cal,SSIM_cal,psnr_,ssim_
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import cosine_similarity


def PSNR(img1, img2):
    b,_,_,_=img1.shape
    img1=np.clip(img1,0,255)
    img2=np.clip(img2,0,255)
    mse = np.mean((img1/ 255. - img2/ 255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
test_dir=r'F:/datasets/feng_experiment/test_data/collection'
test_store=r'F:/sota/MXZ/sots_out1/'#E:/mengxzh/GP/review/view_test/sots_out/'
expname='tpami_work/results/mycollection/'
ensure_dir(test_store)
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#11,
model_dir = r'E:\code2023\LD_Net\trained_models/ots_train_Dehaze_627_256_fLD_model1_epoch1.pk'
device='cuda' if torch.cuda.is_available() else 'cpu'
ckp = torch.load(model_dir,map_location=device)
net = Dehaze(3,3).to('cuda')
net = nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net.eval()
batch_size=1
nyuhaze=False
synTest=True
indoor=False
test_loader=DataLoader(TestData(test_dir,synTest,indoor),batch_size=1)
psnr=[]
ssim=[]
start=time.time()

for j,data_ in enumerate(test_loader):
    with torch.no_grad():
        if synTest:
            test_x,test_img_name= data_
            test_x=test_x.to(device)
            b,c,h,w=test_x.size()
            # print(test_x.shape)
            if h%16!=0:
                h0=h%16#550/16=...6
                if h0%2==0:
                    h1=int(h0/2)
                    test_x=test_x[:,:,h1:(h-h1),:]
                else:
                    h1=h0%2
                    test_x=test_x[:,:,h1:(h+h1-h0),:]
            if w%16!=0:
                w0= w% 16
                if w0%2 == 0:
                    w1 = int(w0 / 2)
                    test_x= test_x[:, :, :, w1:(w - w1)]
                else:
                    w1 = w0 % 2
                    test_x = test_x[:, :, :, w1:(w - w0 + w1)]
            predict_y,_=net(test_x)
        else:
            test_x,test_img_name= data_
            test_x=test_x.to(device)
            b,c,h,w=test_x.size()
            if h%16!=0:
                h0=h%16
                if h0%2==0:
                    h1=int(h0/2)
                    test_x=test_x[:,:,h1:(h-h1),:]
                else:
                    h1=h0%2
                    test_x=test_x[:,:,h1:(h+h1-h0),:]
            if w%16!=0:
                w0=w%16
                if w0%2==0:
                    w1=int(w0/2)
                    test_x=test_x[:,:,:,w1:(w-w1)]
                else:
                    w1=w0%2
                    test_x=test_x[:,:,:,w1:(w-w0+w1)]
            try:
                predict_y,_=net(test_x)
            except:
                continue
    save_image_(predict_y,expname,test_img_name)
    # print(j)
end=time.time()-start
print(end)

