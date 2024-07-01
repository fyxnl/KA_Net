import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import re
import os


class TestData(data.Dataset):
    def __init__(self, img_filename, synTest, indoor):
        super(TestData).__init__()
        self.img_filename = img_filename
        self.img_list = os.listdir(self.img_filename)
        self.len = len(self.img_list)
        self.gt_path = self.img_filename.replace('hazy', 'gt')
        self.synTest = synTest
        self.indoor = indoor

    def __getitem__(self, index):
        img_name = self.img_list[index % self.len]  # 0001_0.8_0.2.jpg
        img_root = os.path.join(self.img_filename, img_name)  # 读取图片
        transform_x = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # transform_x = Compose([ToTensor(), Normalize((0.64, 0.6, 0.58), (0.14, 0.15, 0.152))])
        img = Image.open(img_root).convert('RGB')
        img_trans = transform_x(img)
        return img_trans, img_name
    def __len__(self):
        return self.len