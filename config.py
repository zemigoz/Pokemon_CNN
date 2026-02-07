import os
import torch

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / Path("data")
IMAGE_DIR = DATA_DIR / Path("images")

############ Data Prep
POKEMON_NAMES_CSV = DATA_DIR / Path("pokemon_names.csv")
BAD_REQUEST_CSV = DATA_DIR / Path("bad_requests.csv")

POKEMON_TCG_CARDS_PATH = BASE_DIR / Path("pokemon-tcg-data/cards")

TRAIN_DIR = DATA_DIR / Path("train")
VALIDATION_DIR = DATA_DIR / Path("validation")

EXCLUDE_CARDS = [
    "xyp-XY46", "mcd14-10", "mcd18-10", "xyp-XY68", "mcd14-2", 
    "mcd17-6", "mcd17-7", "mcd18-7", "mcd17-10", "mcd17-9", 
    "mcd18-9", "mcd18-11","mcd15-7","mcd14-3","mcd14-11",
    "mcd14-4","mcd14-2", "mcd18-1", "mcd17-2","mcd14-7",
    "mcd18-3","mcd14-6","xyp-XY39","mcd17-3","mcd15-2",
    "mcd18-6","mcd18-8","mcd15-10","mcd15-9","mcd17-8","mcd15-5",
    "svp-102", "mcd14-5", "mcd15-6", "mcd17-5", "mcd18-4",
    "mcd17-11","mcd17-4","mcd18-12","mcd18-2","mcd15-8",
    "mcd17-1","mcd15-12","mcd18-5","mcd14-8","mcd15-4",
    "mcd14-9","mcd15-3","mcd15-1","hsp-HGSS18","mcd14-1",
    "mcd17-12","mcd15-11"
]

REQUEST_SLEEP_TIMER = 0.1
REQUEST_STREAM_BYTE_SIZE = 8192 * 2048
######################## neural net

WANDB_API_KEY_PATH = BASE_DIR / Path("wandb_api_key.txt")

WANDB_PROJECT_NAME = "tutorial_cnn"

RNG_SEED = 314

FOLDS = 5
VAL_TEST_SPLIT = 0.2

EPOCHS = 10
ALPHA = 1e-3

NUM_WORKERS = min(8, os.cpu_count())
BATCH_SIZE = 32
USE_F16_SCALER = False