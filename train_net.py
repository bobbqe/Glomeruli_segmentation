import logging
import os
import sys
from os import mkdir
import torch.nn.functional as F
import gc
sys.path.append('.')
sys.path.append('..')
from config import cfg
from model.custom_models import build_model
from utils.losses import symmetric_lovasz, dice_coef_loss, sym_lovasz_dice, loss_focal_corrected, loss_sym_lovasz_focal_corrected, sym_lovasz_focal_dice
from fastai.vision.all import *
from data.HuBMAPDataset import HuBMAPDataset
from data.transforms.build_transforms import build_transforms
from utils.metrics import Dice_soft, Dice_th, Dice_th_pred
from utils.prediction_model import Model_pred
from utils.logger import setup_logger
from utils.seed import seed_everything
from utils.helpers import save_img


def train(cfg):
    
    loss_fn = globals()[cfg.SOLVER.LOSS_FUNCTION_NAME]
    seed_everything(cfg.MODEL.SEED)
    transforms = build_transforms(cfg=cfg)
    
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS
    train_folder = cfg.DATASETS.TRAIN_FOLDER

    logger = logging.getLogger("template_model.train")
    logger.info("Start training")
    nfolds = cfg.SOLVER.N_FOLDS
    nworkers = cfg.DATALOADER.NUM_WORKERS
    batch_size = cfg.SOLVER.IMS_PER_BATCH
    lr = slice(cfg.SOLVER.BASE_LR_MIN, cfg.SOLVER.BASE_LR_MAX)

    dice = Dice_th_pred(np.arange(0.2,0.7,0.01))
    for fold in range(nfolds):
        logger.info(f'Fold {fold}')
        model = build_model(cfg, is_train=True)
        suffix_name = f'reduction_{cfg.INPUT.SIZE_REDUCTION_TRAIN}_{cfg.MODEL.ARCHITECTURE}_{cfg.MODEL.ENCODER_NAME}_fold_{fold}'
        ds_t = HuBMAPDataset(data_dir=train_folder, fold=fold, train=True, tfms=transforms, cfg = cfg)
        ds_v = HuBMAPDataset(data_dir=train_folder, fold=fold, train=False, cfg = cfg)
        data = ImageDataLoaders.from_dsets(ds_t, ds_v,bs=batch_size,
                    num_workers=nworkers,pin_memory=True).to(device)
        
        learn = Learner(data, model, model_dir=output_dir, loss_func=loss_fn,
                    metrics=[
                        Dice_soft(),
                        Dice_th()
                        ]).to_fp16()

        learn.fit_one_cycle(
            epochs, lr_max=lr,
            cbs=[
                #SaveModelCallback(monitor='dice_th', comp=np.greater),
                CSVLogger(fname=os.path.join(output_dir, f'history_{suffix_name}.csv'))
                ])
        torch.save(learn.model.state_dict(), os.path.join(output_dir, f'model_{suffix_name}.pth'))
        
        mp = Model_pred(learn.model, learn.dls.loaders[1], cfg)
        with zipfile.ZipFile(os.path.join(output_dir,f'val_masks_tta_{suffix_name}.zip'), 'a') as out:
            for p in progress_bar(mp):
                dice.accumulate(p[0],p[1])
                save_img(p[0],p[2],out)
        gc.collect()
    dices = dice.value.numpy()
    noise_ths = dice.ths
    np.savetxt(os.path.join(output_dir, 'dice_vs_ths.txt'), np.array([dices, noise_ths]).T)


def main():
    
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("template_model", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))

    logger.info("Running with config:\n{}".format(cfg))

    train(cfg)


if __name__ == '__main__':
    main()