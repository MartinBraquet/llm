"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import tiktoken
import torch

from llm.utils import ModelLoader
from model import GPT

# -----------------------------------------------------------------------------
checkpoint_name = 'last'
init_from = 'resume'  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out'  # ignored if init_from is not 'resume'
start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1  # number of samples to draw
max_new_tokens = 100  # number of tokens generated in each sample
temperature = 1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
# examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device {device}')
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

DIR = Path(__file__).parent
out_dir = DIR / 'results' / out_dir

checkpoint = None
# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    model_loader = ModelLoader(
        out_dir=out_dir,
        device=device,
        checkpoint_name=checkpoint_name,
    )
    model = model_loader.load()
    checkpoint = model_loader.checkpoint
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
else:
    raise ValueError(f"Unknown init_from {init_from}")

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
meta_path = False
if init_from == 'resume' and 'config' in checkpoint:
    config = checkpoint['config']
    if 'dataset' in config:  # older results might not have these...
        dataset = config['dataset']
        if os.path.exists(_ := DIR / 'data' / dataset / 'meta.pkl'):
            meta_path = _

if meta_path:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings")
    enc = tiktoken.get_encoding("gpt2")
    encode = partial(enc.encode, allowed_special={"<|endoftext|>"})
    decode = enc.decode

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

print()
print('Output:---------------')
# run generation
with torch.no_grad(), ctx:
    for k in range(num_samples):
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        print(decode(y[0].tolist()))
        print('---------------')
