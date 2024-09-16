import os
from pathlib import Path

import numpy as np
import requests
import tiktoken

from llm.cache.disk import get_disk
from llm.logger import logger

DIR = Path(__file__).parent

DATA_URLS = {
    'prince': 'https://ia601309.us.archive.org/2/items/TheLittlePrince-English/littleprince_djvu.txt',
    'karamazov': 'https://www.gutenberg.org/cache/epub/28054/pg28054.txt',
}


def load_data(name: str, train_ratio: float = 0.97):
    cache = get_disk()
    base_key = f"input_ids_{name}"
    train_ids, val_ids = cache.get(f"{base_key}_train"), cache.get(f"{base_key}_val")
    logger.info(f"loading {name} with train_ratio={train_ratio}")
    if train_ids is None or val_ids is None:
        logger.info(f"downloading {name} data")
        data_url = DATA_URLS[name]
        data = requests.get(data_url).text

        data_dir = DIR / 'data' / name
        os.makedirs(DIR / 'data', exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        input_file_path = data_dir / f'{name}.txt'
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(data)

        n = len(data)
        last_train_index = int(n * train_ratio)
        train_data = data[:last_train_index]
        val_data = data[last_train_index:]

        # encode with tiktoken gpt2 bpe
        enc = tiktoken.get_encoding("gpt2")

        train_ids = enc.encode_ordinary(train_data)
        val_ids = enc.encode_ordinary(val_data)

        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")

        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)

        train_ids.tofile(data_dir / 'train.bin')
        val_ids.tofile(data_dir / 'val.bin')
        # cache[f"{base_key}_train"], cache[f"{base_key}_val"] = train_ids, val_ids

    return train_ids, val_ids
