# Data
DATASET = "TuSimple"
DATA_ROOT = "/kaggle/input/datasets/manideep1108/tusimple/TUSimple/train_set"

# train
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_LANES = 6
OPTIMIZER = 'AdamW'  #['SGD','Adam']

# network
NUM_ROWS = 128
NUM_COLS = 256

CHECKPOINT_DIR = "checkpoints"

LAMBDA_1 = 10 # lambda_vertex
LAMBDA_2 = 1 # lambda_lane