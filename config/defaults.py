from yacs.config import CfgNode as CN
import os
from time import time 

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.ARCHITECTURE = 'UneXt101' #Unet
_C.MODEL.ENCODER_NAME = 'resnext101_32x16d' 
_C.MODEL.ENCODER_WEIGHTS = 'ssl'
_C.MODEL.DEVICE = "gpu:0"
_C.MODEL.NUM_CLASSES = 1
_C.MODEL.SEED = 777

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = 256
_C.INPUT.SIZE_REDUCTION_TRAIN = 16
_C.INPUT.SATURATION_THRESHOLD = 40
_C.INPUT.PIXEL_THRESHOLD = 200*_C.INPUT.SIZE_TRAIN//256

# Size of the image during test
_C.INPUT.SIZE_TEST = 512
_C.INPUT.SIZE_REDUCTION_TEST = 4

# Random probability for augmentation
_C.INPUT.PROB = 0.6
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.63482309, 0.47376275, 0.67814029]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.17405236, 0.23305763, 0.1585981]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.WSI_FOLDER = './data/datasets/WSI_data/'
_C.DATASETS.TRAIN_FOLDER = './data/datasets/data_1/'
_C.DATASETS.LABELS_CSV = './data/datasets/WSI_data/train.csv'

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.LOSS_FUNCTION_NAME = 'loss_sym_lovasz_focal_corrected' 
# Other Loss functions available: 
# symmetric_lovasz, dice_coef_loss, sym_lovasz_dice, loss_focal_corrected, loss_sym_lovasz_focal_corrected, sym_lovasz_focal_dice
_C.SOLVER.MAX_EPOCHS = 100
_C.SOLVER.N_FOLDS = 4
_C.SOLVER.BASE_LR_MIN = 0.000005
_C.SOLVER.BASE_LR_MAX = 0.0005
_C.SOLVER.IMS_PER_BATCH = 32

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 32
_C.TEST.MODEL_WEIGHTS_FOLDER = './model/checkpoints/saved_checkpoints/'
_C.TEST.SIZE = 256
_C.TEST.REDUCE = 32
_C.TEST.MASK_SZ = _C.TEST.REDUCE * _C.TEST.SIZE
_C.TEST.N_BINS = 255
_C.TEST.S_TH = 40  #saturation blancking threshold
_C.TEST.P_TH = 200 * _C.TEST.SIZE//256  #threshold for the minimum number of pixels
_C.TEST.TH = 0.45
_C.TEST.WSI_FOLDER = './data/datasets/WSI_data/'

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
curent_path = os.path.dirname(__file__)
_C.OUTPUT_DIR = os.path.join(curent_path, '..', 'model','checkpoints', f'{int(time())}')
