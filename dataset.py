# dataset.py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset



def load_DCC3240M_frames(root_dir='../raw_data/protov2_110722/0/',
                         orig_shape=(128, 128),
                         emb_shape=(32, 32),
                         n_emb=16,
                         encoder_ker_size=4,
                         target_name='1024x1024_orignal_bmp',
                         amp=1.):
    _real_h = 1024
    _real_w = 1280
    _real_w_start = 130
    _mcv = 255
    data = {}
    for file in os.listdir(root_dir):
        subdir = os.path.join(root_dir, file)
        if os.path.isdir(subdir):
            # Load embedding
            emb_tensor = torch.zeros((1, n_emb, *emb_shape))
            for edx in range(n_emb):
                # Positive embedding
                emb_pos = cv2.imread(os.path.join(subdir, f'{edx}_p.bmp'))
                emb_pos = cv2.cvtColor(emb_pos, cv2.COLOR_BGR2GRAY)
                emb_pos = emb_pos[0:_real_h, 
                    _real_w_start:_real_w_start+_real_h]
                emb_pos_resize = cv2.resize(emb_pos, 
                                        orig_shape, 
                                        interpolation = cv2.INTER_LINEAR)
                emb_pos_resize_crop = emb_pos_resize[
                    orig_shape[0]//2 - emb_shape[0]//2 : orig_shape[0]//2 + emb_shape[0]//2,
                    orig_shape[1]//2 - emb_shape[1]//2 : orig_shape[1]//2 + emb_shape[1]//2,]
                #emb_pos_tensor[edx, 0, :, :] = torch.from_numpy(emb_pos_resize_crop) / _mcv
                # Negative embedding
                emb_neg = cv2.imread(os.path.join(subdir, f'{edx}_n.bmp'))
                emb_neg = cv2.cvtColor(emb_neg, cv2.COLOR_BGR2GRAY)
                emb_neg = emb_neg[0:_real_h, 
                          _real_w_start:_real_w_start+_real_h]
                emb_neg_resize = cv2.resize(emb_neg, 
                                        orig_shape, 
                                        interpolation = cv2.INTER_LINEAR)
                emb_neg_resize_crop = emb_neg_resize[
                    orig_shape[0]//2 - emb_shape[0]//2 : orig_shape[0]//2 + emb_shape[0]//2,
                    orig_shape[1]//2 - emb_shape[1]//2 : orig_shape[1]//2 + emb_shape[1]//2,]
                # Emb tensor compute
                diff = (torch.from_numpy(emb_pos_resize_crop.astype(float)) - \
                        torch.from_numpy(emb_neg_resize_crop.astype(float)))
                diff_scaled = diff / (encoder_ker_size**2 * _mcv / 2)
                diff_scaled = torch.clip(diff_scaled, -1, 1)
                emb_tensor[0, edx, :, :] = diff_scaled
            # Load original
            orig = cv2.imread(os.path.join(subdir, f'{target_name}'))
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            orig = orig[0:_real_h, 
                        _real_w_start:_real_w_start+_real_h]
            orig_resize = cv2.resize(orig, orig_shape, interpolation = cv2.INTER_LINEAR) / _mcv
            orig_tensor = torch.clip(encoder_ker_size**2 * amp * torch.from_numpy(orig_resize).float().view(1, 1, *orig_shape), 0, 1)
            data[file] = (emb_tensor, orig_tensor)
    
    return data


class SimpleBSD500RGB(Dataset):
    
    def __init__(self, 
                 root_dir, 
                 grayscale=True,
                 train=True,
                 shuffle=False,
                 resize_size=128, 
                 rnd_seed=0, 
                 device=torch.device('cpu')):

        if train:
            data_dirs = [root_dir + 'data/images/test/',
                         root_dir + 'data/images/train/']
            self.__n_samples = 400
        else:
            data_dirs = [root_dir + 'data/images/val/',]
            self.__n_samples = 100
        rgb8_mcv = 255
        if grayscale:
            self.__im_dataset = torch.zeros((self.__n_samples, 
                                             1, 
                                             resize_size, resize_size)).to(device)
        else:
            self.__im_dataset = torch.zeros((self.__n_samples, 
                                             3, 
                                             resize_size, resize_size)).to(device)
        idx = 0
        for data_dir in data_dirs:
            im_names = sorted(os.listdir(data_dir))[:-1]
            for fname in im_names:
                img = cv2.imread(data_dir + fname)
                img_lowest_dim = min(img[:, :, 0].shape)
                img = img[:img_lowest_dim, :img_lowest_dim, :]
                img = cv2.resize(img, 
                                 (resize_size, resize_size),
                                 interpolation=cv2.INTER_AREA)
                if grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                t_img = torch.from_numpy(
                    img.transpose(2, 0, 1) / rgb8_mcv
                    ).to(device)
                self.__im_dataset[idx] = t_img
                idx += 1
        assert idx == self.__n_samples

        # shuffle
        if shuffle:
            torch.manual_seed(rnd_seed+6)
            perm = torch.randperm(self.__n_samples)
            self.__im_dataset = self.__im_dataset[perm, :, :, :]


    def __len__(self):
        return self.__n_samples

    def __getitem__(self, idx):
        sample = self.__im_dataset[idx]
        return sample

