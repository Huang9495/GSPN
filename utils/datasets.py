from __future__ import print_function, division

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import cv2
import numpy as np
import os
import json


class PoseDatasetMultiTask(Dataset):
    def __init__(self, imdb_images, mode, args):
        self.imdb_images  = imdb_images
        self.mode= mode
        self.image_files = self._load_image_files()
        self.theta_idx = args.theta_idx
        self.image_size = args.image_size

    def __getitem__(self, index):
        image_file = self.image_files[index]
        x_tensor, y1_tensor, y2_tensor = self._load_data(image_file)

        return  x_tensor, y1_tensor, y2_tensor

    def __len__(self):
        return len(self.image_files)

    def _load_image_files(self):
        image_files_path = os.path.join(self.imdb_images, self.mode + '.txt')
        with open(image_files_path,'r') as f :
            image_files = f.read().splitlines()

        return image_files

    def _load_data(self, image_file):
        image_path = os.path.join(self.imdb_images, image_file)
        basename = os.path.basename(image_file).split('.')[0]
        annos_file = image_path.replace('JPEGImages', 'Annotations').replace('.jpg', '.json')
        assert os.path.exists(annos_file), 'target file exists not'
        lsd_file  = image_path.replace('JPEGImages', 'JPEGImages_lsd')
        assert os.path.exists(lsd_file), 'lsd file exists not'
        im1 = cv2.imread(os.path.join(self.imdb_images, image_file))
        im1 = self._resize(im1, self.image_size)[:, :, ::-1] / 255.
        im2 = cv2.imread(lsd_file, cv2.IMREAD_GRAYSCALE)
        im2 = self._resize(im2, self.image_size)
        im2 = im2[:, :, np.newaxis]/255.
        im_tensor = torch.from_numpy(im1).permute(2, 0, 1).float()
        lsd_tensor = torch.from_numpy(im2).permute(2, 0, 1).float()
        with open(annos_file, 'r') as f:
            em_dict = json.load(f)
        em = (np.asarray(em_dict['euler angle']) + 90) / 180

        pose_tensor = torch.from_numpy(em).float().view(-1)

        return im_tensor, lsd_tensor, pose_tensor

    def _resize(self, img, scale_size):
        scale_factor = scale_size / max(img.shape)
        width = round(img.shape[1] * scale_factor)
        height = round(img.shape[0] * scale_factor)
        dim = (width, height)
        small_boundary = min(width, height)
        start = int((scale_size-small_boundary)/2)
        end = scale_size - start - small_boundary
        # resize image
        img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        # print(img.shape, dim, img_resized.shape)
        if width >= height:
            img_resized = cv2.copyMakeBorder(img_resized, start, end, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif height >= width:
            img_resized = cv2.copyMakeBorder(img_resized, 0, 0, start, end, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return img_resized


