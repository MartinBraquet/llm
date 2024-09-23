import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
import tiktoken

from llm import BASE_DIR
from llm.logger import logger
from llm.utils import str_to_hash

DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / 'data'


@lru_cache
def get_data_paths():
    return json.load(open(DATA_DIR / 'data_paths.json'))


def save_txt_file(data, name):
    os.makedirs(DATA_DIR / name, exist_ok=True)
    input_file_path = DATA_DIR / name / f'{name}.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(data)


def is_url(data_path):
    return str(data_path).startswith('http')


@lru_cache
def get_key(data_path, train_ratio):
    if is_url(data_path):
        s = data_path
    else:
        with open(data_path, 'r') as f:
            s = f.read()
    return str_to_hash(s + str(float(train_ratio)))[:16]


def load_data(
    name: str = 'unnamed',
    train_ratio: float = 0.9,
    data_path: Path = None,
    return_values: bool = False,
):
    logger.info(f"Loading {name} with train_ratio={train_ratio}")
    if data_path is None:
        data_path = get_data_paths()[name]
    key = get_key(data_path, train_ratio=train_ratio)
    data_dir = DATA_DIR / name / key
    path_train, path_val = data_dir / 'train.bin', data_dir / 'val.bin'
    cached = os.path.exists(path_train) and os.path.exists(path_val)
    train_ids = val_ids = None
    if not cached:
        print(f"Downloading {name} data from {data_path}")
        if is_url(data_path):
            data = requests.get(data_path).text
            save_txt_file(data=data, name=name)
        else:
            with open(data_path, 'r') as f:
                data = f.read()

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

        os.makedirs(DATA_DIR / name, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        train_ids.tofile(path_train)
        val_ids.tofile(path_val)

    elif return_values:
        train_ids = np.memmap(path_train, dtype=np.uint16, mode='r')
        val_ids = np.memmap(path_val, dtype=np.uint16, mode='r')

    if return_values:
        return train_ids, val_ids
    else:
        return data_dir
