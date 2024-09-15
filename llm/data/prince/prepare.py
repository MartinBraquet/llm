import os
from pathlib import Path

import numpy as np
import requests
import tiktoken

DIR = Path(__file__).parent
input_file_path = DIR / 'input.txt'
if not os.path.exists(input_file_path):
    data_url = 'https://ia601309.us.archive.org/2/items/TheLittlePrince-English/littleprince_djvu.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_ratio = 0.97
last_train_index = int(n * train_ratio)
train_data = data[:last_train_index]
val_data = data[last_train_index:]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")

train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(DIR / 'train.bin')
val_ids.tofile(DIR / 'val.bin')
