# Constants file


from enum import Enum
import torch

# Categories
CATEGORIES = ['staff']

CATEGORIES_TO_NUM = dict({
  'background': 0,
  'staff': 1,
  'empty-staff': 1,
})
NUM_TO_CATEGORIES = dict({
  0: 'background',
  1: 'staff',
})


# Datasets
DATASET_CAPITAN = 'Capitan'
DATASET_SEILS = 'SEILS'
DATASET_FMT_C = 'FMT_C'

DATASETS = [
  DATASET_CAPITAN,
  DATASET_SEILS,
  DATASET_FMT_C,
]


DEBUG_FLAG = False

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

BBOX_REDIMENSION = 0.8
BBOX_REDIMENSIONED_RECOVER = 1 / BBOX_REDIMENSION

TRAIN_NUM_EPOCHS = 100
TRAIN_PATIENTE = 20

DRIVE_IA_FOLDER = ''

DRIVE_DATASETS_FOLDER = f'datasets'
DRIVE_MODELS_FOLDER = f'models'
DRIVE_LOGS_FOLDER = f'logs'
DRIVE_IMG_FOLDER = f'img'

DRIVE_TRAIN_LOGS_FOLDER = f'{DRIVE_LOGS_FOLDER}/train'
DRIVE_VAL_LOGS_FOLDER = f'{DRIVE_LOGS_FOLDER}/val'
DRIVE_TEST_LOGS_FOLDER = f'{DRIVE_LOGS_FOLDER}/test'

DRIVE_VAL_IMG_FOLDER = f'{DRIVE_IMG_FOLDER}/val'
DRIVE_TEST_IMG_FOLDER = f'{DRIVE_IMG_FOLDER}/test'

SAE_IMAGE_SIZE =  (512, 512)

BIN_UMBRALS = [i/100 for i in range(10, 91, 5)]
DROPOUT_VALUES = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

class PredictionsCombinationType(Enum):
    NONE = ''
    MEAN = 'mean'
    MAX  = 'max'
    VOTES = 'votes'

PREDICTIONS_COMBIATION_TYPES = [
  PredictionsCombinationType.MEAN,
  PredictionsCombinationType.MAX,
  PredictionsCombinationType.VOTES
  ]
PREDICTIONS_COMBINATION_QUANTITIES = [1, 3, 7, 15, 31, 63]

class ForwardParameters:
  def __init__(
      self,
      dataset_name: str,
      uses_redimension_vertical: bool = True,
      uses_redimension_horizontal: bool = True,
      bin_umbral: float = 0.5,
      train_dropout: float = 0.0,
      val_dropout: float = 0.0,
      times_pass_model: int = 1,
      type_combination: PredictionsCombinationType = PredictionsCombinationType.NONE
    ):
      self.dataset_name = dataset_name
      self.uses_redimension_vertical = uses_redimension_vertical
      self.uses_redimension_horizontal = uses_redimension_horizontal
      self.bin_umbral = bin_umbral
      self.train_dropout = train_dropout
      self.val_dropout = val_dropout
      self.times_pass_model = times_pass_model
      self.type_combination = type_combination

def init():
   import random
   import numpy as np 
   import torch 
   import os

   seed_value = 122
   os.environ['PYTHONHASHSEED'] = str(seed_value)
   random.seed(seed_value)
   np.random.seed(seed_value)
   #tf.random.set_seed(seed_value)
   torch.manual_seed(seed_value)

