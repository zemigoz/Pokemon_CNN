import time
import os
import json
import re
import unicodedata
import pandas as pd

from pathlib import Path
from collections import defaultdict

import requests

from config import *
from train_val_split import train_val_split

################################
# Main Walk
################################
def sanitize(string: str) -> str:
    replacement_map = {
        'A©': 'e',
        'Æ': 'AE',
        'æ': 'ae',
        'ß': 'ss',
        "I´": ""
    }

    pokemon_name = unicodedata.normalize('NFD', string)

    pokemon_name = ''.join(c for c in pokemon_name if unicodedata.category(c) != 'Mn') #gets rid of accents
    pattern = re.compile("|".join(re.escape(k) for k in replacement_map.keys()))
    pokemon_name = pattern.sub(lambda m: replacement_map[m.group(0)], pokemon_name) #SO MUCH WORK JUST TO MAKE IT POKEMON AND NOT POKACMON, it aint an AC unit

    pokemon_name = re.sub(r"[^a-zA-Z0-9_-\u00E9]", "_", pokemon_name)
    pokemon_name = re.sub(r"_+", "_", pokemon_name)                  # collapse multiple underscores
    pokemon_name = pokemon_name.strip("_")


    return pokemon_name

def main():
    # name_df = pd.read_csv(POKEMON_NAMES_DIR)
    # name_set = set(name_df["Name"])
    bad_requests = []

    # for set in POKEMON_TCG_CARDS_PATH.rglob("smp.json"):
    for set in POKEMON_TCG_CARDS_PATH.rglob("*.json"):
        print(f"On set {set}")

        card_data = []
        with open(set, "r", encoding="utf-8") as f:
            card_data = json.load(f)

        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        pokemon_counts = defaultdict(int)
        length = len(card_data)

        for i, card in enumerate(card_data, start=1):
        # for i,card in enumerate(card_data[107:109],start=1):
            if card['id'] in EXCLUDE_CARDS:
                continue

            print(f"Working on image {i}/{length} in set {set.stem}")
            pokemon_name = sanitize(card['name'])
            # print(pokemon_name)

            if re.search(r"Unown", pokemon_name):
                pokemon_name = re.sub(r"_[\w]$", "", pokemon_name)
            elif re.search(r"Ho-oh", pokemon_name, re.IGNORECASE):
                pokemon_name = "Ho-oh"
            elif re.search(r"Necrozma", pokemon_name, re.IGNORECASE):
                pokemon_name = "Necrozma"


            leading_pokemon_tags_patterns = r"""
            \A
            (?:
            Erika_s|Giovanni_s|Holon_s|Koga_s|Hop_s|
            Rocket_s|Lt_Surge_s|Team_Magma_s|Team_Plama|Team_Rocket_s|
            Sabrina_s|Blaine_s|Steven_s|Arven_s|Cynthia_s|Brock_s|
            Lance_s|Ethan_s|Larry_s|Iono_s|Lillie_s|Marnie_s|Misty_s|
            N_s|Team_Aqua_s|Ash_s|Ash|

            Radiant_Hisuian|Alolan|Galarian|Hisuian|Paldean|Radiant|Dark|Light|Mega|
            
            s|Detective|Surfing|Cool|Snow_cloud|Shining|M|
            Single_Strike|Rapid_Strike|Bloodmoon|Dusk_Mane|
            Dawn_Wings|Ultra_Necrozma|Fan|Flying|Wash|
            Rain|Ice_Rider|Shadow_Rider|Sunny|East_Sea|West_Sea|
            Wellspring_Mask|Teal_Mask|Cornerstone_Mask|
            Hearthflame_Mask|Mow|Heat|Special_Delivery|
            Armored|Primal)
            (?:-|_)
            """
            leading_pokemon_tags_patterns = re.compile(leading_pokemon_tags_patterns, re.VERBOSE)

            trailing_pokemon_tags_patterns = r"""
            (?:-|_)
            (?:GX|EX|V|VMAX|LV_X|C_LV_X|C|G_LV_X|ex|G|a|i|ex_i|a_i|C|
            GL|E4|FB|X|Y|E4_LV_X|FB_LV_X|X_ex|Y_ex|GL_LV_X|

            Sandy_Cloak|Trash_Cloak|Plant_Cloak|Attack_Forme|
            Defense_Forme|Normal_Forme|Speed_Forme|Rain_Form|Rainy_Form|
            Snowy_Form|
            
            LEGEND|Snow_Cloud_Form|Sunny_Form|GL|East_Sea|West_Sea|
            on_the_ball|V_Union|V_UNION|VSTAR|BREAK|with_Grey_Felt_Hat|
            Alolan_Raichu_GX|on_the_Ball|Bros)
            \Z
            """
            trailing_pokemon_tags_patterns = re.compile(trailing_pokemon_tags_patterns, re.VERBOSE)

            pokemon_name_cleaned = leading_pokemon_tags_patterns.sub("", pokemon_name)
            # print(pokemon_name_cleaned)
            pokemon_name_cleaned = trailing_pokemon_tags_patterns.sub("", pokemon_name_cleaned)
            # print(pokemon_name_cleaned)
            
            img_url = card['images']['large']

            out_dir = IMAGE_DIR / pokemon_name_cleaned
            out_dir.mkdir(parents=True, exist_ok=True)

            idx = pokemon_counts[pokemon_name_cleaned]
            pokemon_counts[pokemon_name] +=1
            
            sanitized_card = sanitize(card['id'])
            # print(sanitized_card)

            img_path = out_dir / Path(f"{sanitized_card}-{pokemon_name}-{idx}.png")
            if img_path.exists():
                continue #already downloaded

            response = requests.get(url=img_url, stream=True, timeout=30) # 30 second timeout
            if 200 < response.status_code or 300 <= response.status_code:
                bad_requests.append(img_url)

            with open(img_path, "wb") as f:
                for chunk in response.iter_content(REQUEST_STREAM_BYTE_SIZE):
                    if chunk:
                        f.write(chunk)

            time.sleep(REQUEST_SLEEP_TIMER)

        bad_requests_df = pd.DataFrame(bad_requests, columns=["url"])
        bad_requests_df.to_csv(BAD_REQUEST_CSV, index=False)

    # train_val_split(make_folders=True)

if __name__ == '__main__':
    start_time = time.time()
    print(f"Started program")
    main()
    print(f'Time to run whole program: {time.time() - start_time}')
