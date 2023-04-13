import torch
from torch.utils.data import Dataset
import cv2
import os
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import sys
sys.path.append('..')

def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))

class HuBMAPDataset(Dataset):
    def __init__(self, data_dir,fold=0, train=True, tfms=None, cfg = None):
        self.cfg = cfg
        nfolds = cfg.SOLVER.N_FOLDS
        folders = [dir for dir in os.listdir(data_dir) if (os.path.isdir(os.path.join(data_dir, dir)) and (('images' in dir) or ('masks' in dir)))]
        
        if len(folders) != 2:
            raise Exception('Train or masks folder were not found')
        if 'images' in folders[0]:
            train_folder = folders[0]
            masks_folder = folders[1]
        else:
            train_folder = folders[1]
            masks_folder = folders[0]

        train_folder = os.path.join(data_dir, train_folder)
        masks_folder = os.path.join(data_dir, masks_folder)
        df_lab = pd.read_csv(cfg.DATASETS.LABELS_CSV)
        ids = df_lab.id.values
        kf = KFold(n_splits=nfolds,random_state=cfg.MODEL.SEED, shuffle=True)
        ids = list(set(ids[list(kf.split(ids))[fold][0 if train else 1]]))
        self.fnames = [fname for fname in os.listdir(train_folder) if fname.split('_')[0] in ids or 
                       (fname.split('_')[0]+'_'+fname.split('_')[1]) in ids]
        self.train_folder = train_folder
        self.masks_folder = masks_folder
        self.train = train
        self.tfms = tfms
        
    def __len__(self):
        return len(self.fnames)
    
    def load_img_mask(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.train_folder,fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.masks_folder,fname),cv2.IMREAD_GRAYSCALE)
        if self.tfms is not None:
            augmented = self.tfms(image=img.astype(np.uint8),mask=mask.astype(np.uint8))
            img,mask = augmented['image'],augmented['mask']
        return img, mask
    
    def __getitem__(self, idx):
        img, mask = self.load_img_mask(idx)
        return img2tensor((img/255.0 - np.array(self.cfg.INPUT.PIXEL_MEAN))/np.array(self.cfg.INPUT.PIXEL_STD)),img2tensor(mask)

class HuBMAPTestDataset(Dataset):
    def __init__(self, idxs, cfg):
        self.fnames = idxs
        self.mean = cfg.INPUT.PIXEL_MEAN
        self.std = cfg.INPUT.PIXEL_STD
        
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        im = cv2.imread("tmp/%d.png" %(self.fnames[idx],))
        return img2tensor((im/255.0 - self.mean)/self.std)