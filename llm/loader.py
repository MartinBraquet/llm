import json
import os
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
import tiktoken

from llm import BASE_DIR
from llm.logger import logger
from llm.utils import list_to_hash, make_dir

DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / 'data'


@lru_cache
def get_data_paths():
    with open(DATA_DIR / 'data_paths.json') as f:
        return json.load(f)


def save_txt_file(data, data_dir):
    input_file_path = data_dir / f'input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(data)


def is_url(data_path):
    return str(data_path).startswith('http')


def is_path(data_path):
    return os.path.exists(data_path) or is_url(data_path)


@lru_cache
def get_key(
    data_path,
    train_ratio,
    encoding,
):
    if is_url(data_path):
        s = data_path
    else:
        with open(data_path, 'r') as f:
            s = f.read()
    return list_to_hash([s, str(float(train_ratio)), encoding])[:16]


def load_data(
    data_path: Path | str,
    name: str = None,
    train_ratio: float = 0.9,
    return_values: bool = False,
    encoding: str = 'gpt2',
):
    """
    :param data_path: path to data or key in data_paths.json
    :param name: dataset name (used only for the directory name where to store the data)
    :param train_ratio: what fraction of the data to use for training
    :param return_values: if True, return the values. if False, return the full data path
    :param encoding: 'gpt2' or 'char'
    """
    assert data_path is not None, "data path must be provided"
    name = name or 'unnamed'
    logger.info(f"Loading {data_path} with train_ratio={train_ratio}")
    if not is_path(data_path):
        name = data_path
        data_path = get_data_paths().get(data_path)
        if data_path is None:
            raise ValueError(f"Unknown data_path: {data_path}. ")
    key = get_key(
        data_path,
        train_ratio=train_ratio,
        encoding=encoding,
    )
    data_dir = Path('data') / name / key
    path_train, path_val = data_dir / 'train.bin', data_dir / 'val.bin'
    cached = os.path.exists(path_train) and os.path.exists(path_val)
    train_ids = val_ids = None
    if not cached:
        make_dir(data_dir)
        print(f"Downloading data from {data_path} ({name=})")
        if is_url(data_path):
            data = requests.get(data_path).text
            save_txt_file(data=data, data_dir=data_dir)
        else:
            with open(data_path, 'r') as f:
                data = f.read()

        n = len(data)
        last_train_index = int(n * train_ratio)
        train_data = data[:last_train_index]
        val_data = data[last_train_index:]

        if encoding == 'gpt2':
            enc = tiktoken.get_encoding("gpt2")
            encode = enc.encode_ordinary
        elif encoding == 'char':
            chars = sorted(list(set(data)))
            vocab_size = len(chars)
            print("all the unique characters:", ''.join(chars))
            print(f"vocab size: {vocab_size:,}")
            stoi = {ch: i for i, ch in enumerate(chars)}
            itos = {i: ch for i, ch in enumerate(chars)}
            with open(data_dir / 'meta.pkl', 'wb') as f:
                pickle.dump({'stoi': stoi, 'itos': itos}, f)
            encode = lambda s: [stoi[c] for c in s]
        else:
            raise ValueError(f"Unknown encoding {encoding}, use 'gpt2' or 'char'")

        train_ids = encode(train_data)
        val_ids = encode(val_data)

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
