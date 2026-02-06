from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / Path("data")
IMAGE_DIR = DATA_DIR / Path("images")
POKEMON_TCG_CARDS_PATH = BASE_DIR / Path("pokemon-tcg-data/cards")
WANDB_API_KEY_PATH = BASE_DIR / Path("wandb_api_key.txt")

REQUEST_SLEEP_TIMER = 0.1
REQUEST_STREAM_BYTE_SIZE = 8192

WANDB_PROJECT_NAME = "tutorial_cnn"

RNG_SEED = 314

FOLDS = 5

EPOCHS = 10
BATCH_SIZE = 64
ALPHA = 1e-3