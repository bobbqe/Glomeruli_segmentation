import os
from os import mkdir
import sys
sys.path.append('.')
sys.path.append('..')
from config import cfg
from utils.losses import symmetric_lovasz, dice_coef_loss, sym_lovasz_dice, loss_focal_corrected, loss_sym_lovasz_focal_corrected, sym_lovasz_focal_dice
from utils.logger import setup_logger
import numpy as np
import zipfile
import tifffile as tiff
import pandas as pd
import cv2
from utils.helpers import enc2mask
from tqdm import tqdm
import os
import sys
sys.path.append('.')
sys.path.append('..')


def create(cfg):
    suffix = f'images_{cfg.INPUT.SIZE_TRAIN}_red{cfg.INPUT.SIZE_REDUCTION_TRAIN}/'
    DATA = cfg.DATASETS.WSI_FOLDER
    MASKS = cfg.DATASETS.LABELS_CSV
    OUT_TRAIN = os.path.join(cfg.DATASETS.TRAIN_FOLDER, suffix)
    if not os.path.exists(OUT_TRAIN):
        os.makedirs(OUT_TRAIN)
    OUT_MASKS = os.path.join(cfg.DATASETS.TRAIN_FOLDER, suffix.replace('images','masks'))
    if not os.path.exists(OUT_MASKS):
        os.makedirs(OUT_MASKS)
    s_reduce = cfg.INPUT.SIZE_REDUCTION_TRAIN
    sz = cfg.INPUT.SIZE_TRAIN
    
    s_th = cfg.INPUT.SATURATION_THRESHOLD
    p_th = cfg.INPUT.PIXEL_THRESHOLD


    df_masks = pd.read_csv(MASKS).set_index('id')

    x_tot, x2_tot = [], []
    
    for index, encs in tqdm(df_masks.iterrows(),total=len(df_masks)):
            # print(encs)
        # print(preds)
        #read image and generate the mask
        img = tiff.imread(os.path.join(DATA, index+'.tiff'))
        if len(img.shape) == 5:
            img = np.transpose(img.squeeze(), (1, 2, 0))
        if img.shape[2] != 3:
            img = np.transpose(img, (1, 2, 0))
        mask = enc2mask(encs, (img.shape[1], img.shape[0]))
        # print(mask.sum())

        #add padding to make the image dividable into tiles
        shape = img.shape
        pad0 = (s_reduce*sz - shape[0]%(s_reduce*sz))%(s_reduce*sz)
        pad1 = (s_reduce*sz - shape[1]%(s_reduce*sz))%(s_reduce*sz)
        # print(img.shape, pad0, pad1)
        img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                    constant_values=0)
        mask = np.pad(mask,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2]],
                    constant_values=0)

        #split image and mask into tiles using the reshape+transpose trick
        img = cv2.resize(img,(img.shape[1]//s_reduce,img.shape[0]//s_reduce),
                        interpolation = cv2.INTER_AREA)
        img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
        img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)

        mask = cv2.resize(mask,(mask.shape[1]//s_reduce,mask.shape[0]//s_reduce),
                        interpolation = cv2.INTER_NEAREST)
        mask = mask.reshape(mask.shape[0]//sz,sz,mask.shape[1]//sz,sz)
        mask = mask.transpose(0,2,1,3).reshape(-1,sz,sz)

        #write data
        for i,(im,m) in enumerate(zip(img,mask)):
            #remove black or gray images based on saturation check
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            if (s>s_th).sum() <= p_th or im.sum() <= p_th: continue

            x_tot.append((im/255.0).reshape(-1,3).mean(0))
            x2_tot.append(((im/255.0)**2).reshape(-1,3).mean(0))

            
            cv2.imwrite(os.path.join(OUT_TRAIN,f'{index}_{i}red{s_reduce}.png'), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(OUT_MASKS,f'{index}_{i}red{s_reduce}.png'), m)

    #image stats
    img_avr = np.array(x_tot).mean(0)
    img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
    print('mean:', img_avr, ', std:', img_std)

    return (img_avr, img_std)


def main():
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("template_model", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info("Running with config:\n{}".format(cfg))

    create(cfg)


if __name__ == '__main__':
    main()