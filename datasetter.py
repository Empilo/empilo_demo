import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random


class FrameDataset(Dataset):
    def __init__(self, directories, mode="shuffle"):
        self.directories = directories
        self.tf = transforms.ToTensor()
        self.mode = mode
        self.file_list, self.len_list = self.get_file_list()

    def get_file_list(self):
        file_list = []
        len_list = []
        for directory in self.directories:
            list = os.listdir(directory)
            list.sort()
            file_list.append(list)
            len_list.append(len(list))
        print(f" Dataset Size (# of images, videos): {sum(len_list)} / {len(len_list)}")
        return file_list, len_list

    def get_frame(self, idx, num=None):
        if num is None:
            cumulative_sum = 0
            num = -1
            while cumulative_sum <= idx and num < len(self.directories):
                num += 1
                cumulative_sum += self.len_list[num]

            img_name = os.path.join(self.directories[num], self.file_list[num][idx - cumulative_sum])

        else:
            img_name = os.path.join(self.directories[num], self.file_list[num][idx])

        image = Image.open(img_name)
        image_np = np.array(image)
        return image_np, num

    def __len__(self):
        return sum(self.len_list)

    def __getitem__(self, idx):
        idx_ori = idx
        image_ori, num_ori = self.get_frame(idx_ori)

        idx_inj_ori = random.randint(0, self.len_list[num_ori] - 1)
        inj_ori, _ = self.get_frame(idx_inj_ori, num_ori)

        idx_cross = (idx + 3) % sum(self.len_list)
        image_cross, num_cross = self.get_frame(idx_cross)

        idx_inj_cross = random.randint(0, self.len_list[num_cross] - 1)
        inj_cross, _ = self.get_frame(idx_inj_cross, num_cross)

        return image_ori, inj_ori, image_cross, inj_cross, (num_ori, num_cross), (idx_ori, idx_cross)
