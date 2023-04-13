import argparse
import os
import sys
from os import mkdir
import torch
from tqdm import tqdm
sys.path.append('.')
sys.path.append('..')
from config import cfg
from utils.logger import setup_logger
import gc
import numpy as np
from PIL import Image
import cv2
import tifffile as tiff
from utils.prediction_model import Model_pred_test
from model.custom_models import build_model
import time
from data.HuBMAPDataset import HuBMAPTestDataset
from fastai.vision.all import *
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

def main():
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("template_model", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info("Running with config:\n{}".format(cfg))

    models_pred = []
    models_pths = [f for f in os.listdir(cfg.TEST.MODEL_WEIGHTS_FOLDER) if '.pth' in f]
    
    device = cfg.MODEL.DEVICE
    for path in models_pths:
        state_dict = torch.load(os.path.join(cfg.TEST.MODEL_WEIGHTS_FOLDER,path), map_location=torch.device('cpu'))
        model = build_model(cfg)
        model.load_state_dict(state_dict)
        model.float()
        model.eval()
        model.to(device)
        models_pred.append(model)
        del state_dict

    all_test_tiffs = list(set([f.replace('.tiff','') for f in os.listdir(cfg.TEST.WSI_FOLDER) if '.tiff' in f]))
    names,preds = [],[]
    for idx in tqdm(all_test_tiffs, total=len(all_test_tiffs)):
        print("Processing image", idx)
        rle = make_predictions(idx, models_pred)
        names.append(idx)
        preds.append(rle)

    df = pd.DataFrame({'id':names,'predicted':preds})
    df.to_csv('output.csv',index=False)    

def make_predictions(idx, models_pred):
    mask_sz = cfg.TEST.MASK_SZ
    n_bins = cfg.TEST.N_BINS
    TH = cfg.TEST.TH
    p0_list = [0., 0.5]  # to overlap the tiles during inference
    p1_list = [0., 0.5]  # and improve edge artefacts
#     p0_list = [0.]
#     p1_list = [0.]
    init_shape = get_mask_tiles(idx, p0_list, p1_list, models_pred)
    mask = torch.zeros(*init_shape[:2], dtype=torch.uint8)
    x = 0
    while x < init_shape[0]:
        y = 0
        while y < init_shape[1]:
            mask_tile = 0.
            for p0 in p0_list:
                for p1 in p1_list:
                    tile_path = "%s_%d_%d_%s_%s.png" %(idx, x, y, str(p0), str(p1))
                    mask_tile += torch.tensor(np.asarray(Image.open(tile_path), dtype=int))
                    os.remove(tile_path)
            mask[x:x+mask_sz,y:y+mask_sz] = mask_tile>int(n_bins*len(p0_list)*len(p1_list)*TH)
            y += mask_sz
        x += mask_sz
    
    print("  > Converting to RLE...")
    rle = rle_encode_less_memory(mask.numpy())
    del mask
    return rle

def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def tile_resize_save(img, img_id, tile_sz, reduce=1):
    x = 0
    while x < img.shape[0]:
        y = 0
        while y < img.shape[1]:
            img_tile = img[x:x+tile_sz,y:y+tile_sz]
            if reduce > 1:
                new_dim = (img_tile.shape[1]//reduce,img_tile.shape[0]//reduce)
                img_tile = cv2.resize(img_tile, new_dim, interpolation = cv2.INTER_AREA)
            save_path = "%s_%d_%d.png" %(img_id, x//reduce, y//reduce)
            Image.fromarray(img_tile).save(save_path)
            y += tile_sz
        x += tile_sz
    final_x = ((x-tile_sz)//tile_sz)*(tile_sz//reduce) + img_tile.shape[0]
    final_y = ((y-tile_sz)//tile_sz)*(tile_sz//reduce) + img_tile.shape[1]
    return (final_x, final_y, 3)

def reconstruct_img(img_id, tile_sz, shape):
    img = np.zeros(shape, dtype=np.uint8)
    print("Reconstructed image:", shape)
    x = 0
    while x < shape[0]:
        y = 0
        while y < shape[1]:
            tile_path = "%s_%d_%d.png" %(img_id, x, y)
            img_tile = np.asarray(Image.open(tile_path))
            img[x:x+tile_sz,y:y+tile_sz] = img_tile
            os.remove(tile_path)
            y += tile_sz
        x += tile_sz
    return img

def load_image1(filename):
    img_id = filename.split("/")[-1].split(".")[0]

    image = tiff.imread(filename)
    if image.shape[0] == 3 and image.ndim == 3:
        image = image.transpose(1, 2, 0)
    elif image.shape[2] == 3 and image.ndim == 5 :
        image = np.transpose(image.squeeze(), (1, 2, 0))
    image = np.ascontiguousarray(image)

    print("Initial size:", image.shape)
    return image

def load_resize(idx, reduce, DATA):
    sz = cfg.TEST.SIZE
    img = load_image1(os.path.join(DATA,idx+'.tiff'))
    init_shape = img.shape
    shape = tile_resize_save(img, idx, (sz*reduce*reduce), reduce=reduce)
    img = reconstruct_img(idx, (sz*reduce*reduce)//reduce, shape)

    return img, init_shape

def get_mask_tiles(idx, p0_list, p1_list, models_pred):
    reduce = cfg.TEST.REDUCE
    img, init_shape = load_resize(idx, reduce, cfg.TEST.WSI_FOLDER)
    for p0 in p0_list:
        for p1 in p1_list:
            make_one_prediction(img, idx, init_shape, p0, p1, models_pred)
    return init_shape

def make_one_prediction(img, idx, img_shape, p0, p1, models_pred):
    s_th = cfg.TEST.S_TH
    p_th = cfg.TEST.P_TH
    #add padding to make the image dividable into tiles
    start = time.time()
    print("  > Adding padding to make the image dividable into tiles...")
    reduce = cfg.TEST.REDUCE
    sz = cfg.TEST.SIZE
    pad0 = (reduce*sz - img_shape[0]%(reduce*sz))%(reduce*sz)
    pad1 = (reduce*sz - img_shape[1]%(reduce*sz))%(reduce*sz)

    print("  > Before reduction:", img_shape)
    print("  > After reduction:", img.shape)
    pad0_ = sz - img.shape[0]%sz
    pad0_lr = [(pad0_//2 + int(sz*p0)),pad0_+sz-(pad0_//2 + int(sz*p0))]
    pad1_ = sz - img.shape[1]%sz
    pad1_lr = [(pad1_//2 + int(sz*p1)),pad1_+sz-(pad1_//2 + int(sz*p1))]

    img = np.pad(img,[pad0_lr, pad1_lr,[0,0]],constant_values=0)
    print("  > After padding:", img.shape, "Time =", time.time() - start, "s")

    #split image into tiles using the reshape+transpose trick
    start = time.time()
    print("  > Splitting image into tiles...")
    img_shape_p = img.shape
    assert not img.shape[0]%sz
    assert not img.shape[1]%sz
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    print("  > Splitting done! Time =", time.time() - start)

    #select tiles for running the model
    start = time.time()
    print("  > Selecting tiles...")
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    idxs = []
    for i,im in enumerate(img):
        #remove black or gray images based on saturation check
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if (s>s_th).sum() <= p_th or im.sum() <= p_th: continue
                
        cv2.imwrite("tmp/%d.png" %(i,), im)
        idxs.append(i)
    #tile dataset

    NUM_WORKERS = cfg.DATALOADER.NUM_WORKERS
    bs = cfg.TEST.IMS_PER_BATCH
    ds = HuBMAPTestDataset(idxs, cfg)
    dl = DataLoader(ds,bs,num_workers=NUM_WORKERS,shuffle=False,pin_memory=True)
    mp = Model_pred_test(models_pred,dl,cfg, models_pred)
    mask_sz = cfg.TEST.MASK_SZ
    print("  > Tiles selected! Time =", time.time() - start)

    #generate masks
    start = time.time()
    print("  > Generating masks...")
    mask = torch.zeros(img.shape[0],sz*reduce,sz*reduce,dtype=torch.uint8)
    for i,p in tqdm(zip(idxs,iter(mp))): 
        mask[i] = p.squeeze(-1)
    print("  > Masks generated! Time =", time.time() - start)

    #reshape tiled masks into a single mask and crop padding
    start = time.time()
    print("  > Merge tiled masks into one mask and crop padding...")
    mask = mask.view(img_shape_p[0]//sz,img_shape_p[1]//sz,sz*reduce,sz*reduce).\
        permute(0,2,1,3).reshape(img_shape_p[0]*reduce,img_shape_p[1]*reduce)
    mask = mask[pad0//2 + int(sz*reduce*p0):-(pad0+sz*reduce-pad0//2 - int(sz*reduce*p0)) if pad0 > 0 else img_shape_p[0]*reduce,\
        (pad1//2 + int(sz*reduce*p1)):-(pad1+sz*reduce-pad1//2 - int(sz*reduce*p1)) if pad1 > 0 else img_shape_p[1]*reduce]
    print("  > Mask created! Shape =", mask.shape,"Time =", time.time() - start)
    gc.collect()
    shutil.rmtree('tmp')
    x = 0
    start = time.time()
    print("  > Saving tiles in HDD memory...")
    while x < mask.shape[0]:
        y = 0
        while y < mask.shape[1]:
            mask_tile = mask[x:x+mask_sz,y:y+mask_sz].numpy()
            save_path = "%s_%d_%d_%s_%s.png" %(idx, x, y, str(p0), str(p1))
            Image.fromarray(mask_tile).save(save_path)
            y += mask_sz
        x += mask_sz
    print("Tiles saved! Time =", time.time() - start)

if __name__ == '__main__':
    main()