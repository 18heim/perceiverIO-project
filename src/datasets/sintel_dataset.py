import torch
import torch.utils.data as data

import os, math, random
from os.path import *
import numpy as np

from glob import glob
from datasets.sintel_utils import read_gen

from typing import Optional
from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import re


class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]

LIST_VAL_MAP = ['ambush_2', 'ambush_6','bamboo_2', 'cave_4', 'market_6', 'temple_2']
LIST_TRAIN_MAP = ['alley_2', 'ambush_4', 'ambush_5', 'ambush_6', 'ambush_7', 'bamboo_1',
 'bandage_1', 'bandage_2', 'cave_2', 'market_2', 'market_5', 'mountain_1', 'shaman_2',
 'shaman_3', 'sleeping_1', 'sleeping_2', 'temple_1']       

class MpiSintel(data.Dataset):
    def __init__(self, args, is_cropped = False, data_dir = '', map_list = LIST_TRAIN_MAP,
                 dstype = 'clean', replicates = 1, channels_last = True):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates
        self.channels_last = channels_last

        flow_root = join(data_dir, 'flow')
        image_root = join(data_dir, dstype)

        file_list = sorted(glob(join(flow_root, '*/*.flo')))

        self.flow_list = []
        self.image_list = []

        for file in file_list:
            if 'test' in file:
                # print file
                continue

            fbase = file[len(flow_root)+1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])

            img1 = join(image_root, fprefix + "%04d"%(fnum+0) + '.png')
            img2 = join(image_root, fprefix + "%04d"%(fnum+1) + '.png')

            if not isfile(img1) or not isfile(img2) or not isfile(file):
                continue

            if re.search('^([^_\d]+)\_\d', fprefix).group() in map_list:
                self.image_list += [[img1, img2]]
                self.flow_list += [file]

        self.size = len(self.image_list)

        self.frame_size = read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
            self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
            self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

        args.inference_size = self.render_size

        assert (len(self.image_list) == len(self.flow_list))

    def __getitem__(self, index):

        index = index % self.size

        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])

        flow = read_gen(self.flow_list[index])

        images = [img1, img2]
        image_size = img1.shape[:2]

        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3,0,1,2)
        flow = flow.transpose(2,0,1)

        images = torch.from_numpy(images.astype(np.float32))
        # channels last
        if self.channels_last:
            images = images.permute(1, 2, 3, 0).contiguous()
        flow = torch.from_numpy(flow.astype(np.float32))
        return images[0], flow

    def __len__(self):
        return self.size * self.replicates

class SintelDataModule(pl.LightningDataModule):
    def __init__(self,
                 args,
                 data_dir: Optional[Path] = None,
                 batch_size: int = 16,
                 is_cropped = False,
                 channels_last = True):
      train_dir = join(data_dir, 'training')
      self.training_set = MpiSintel(args, is_cropped, train_dir, map_list=LIST_TRAIN_MAP, dstype='final', channels_last=channels_last)
      self.validation_set = MpiSintel(args, is_cropped, train_dir, map_list=LIST_TRAIN_MAP, dstype='clean', channels_last=channels_last)
      self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.training_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation_set, batch_size=self.batch_size)