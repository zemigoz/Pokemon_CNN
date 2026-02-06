import time
import os
import json
import requests
import re
import pandas as pd

from pathlib import Path
from collections import defaultdict

from pokemontcgsdk import Card
from pokemontcgsdk import Set
from pokemontcgsdk import Type
from pokemontcgsdk import Supertype
from pokemontcgsdk import Subtype
from pokemontcgsdk import Rarity

from config import *

################################
# Main Walk
################################
def main():
    for set in POKEMON_TCG_CARDS_PATH.rglob("*.json"):
    # for set in POKEMON_TCG_CARDS_PATH.rglob("*.json"):
        card_data = []
        with open(set, "r") as f:
            card_data = json.load(f)

        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        pokemon_counts = defaultdict(int)

        for card in card_data:
            pokemon_name = re.sub(r"[^a-zA-Z0-9_-]", "_", card['name'])
            img_url = card['images']['large']

            out_dir = IMAGE_DIR / pokemon_name
            out_dir.mkdir(parents=True, exist_ok=True)

            idx = pokemon_counts[pokemon_name]
            pokemon_counts[pokemon_name] +=1

            img_path = out_dir / Path(f"{card['id']}-{pokemon_name}-{idx}.png")
            if img_path.exists():
                continue #already downloaded

            req = requests.get(url=img_url, stream=True, timeout=30) # 30 second timeout
            req.raise_for_status()

            with open(img_path, "wb") as f:
                for chunk in req.iter_content(REQUEST_STREAM_BYTE_SIZE):
                    if chunk:
                        f.write(chunk)

            time.sleep(REQUEST_SLEEP_TIMER)


if __name__ == '__main__':
    start_time = time.time()
    print(f"Started program")
    main()
    print(f'Time to run whole program: {time.time() - start_time}')
