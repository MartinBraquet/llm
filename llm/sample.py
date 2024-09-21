import os
import pickle
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import tiktoken
import torch

from llm.configurator import FileConfig, get_relevant_config
from llm.model import GPT
from llm.utils import ModelLoader, unbox, DataclassUtils, get_default_device

DIR = Path(__file__).parent


@dataclass
class ModelConfig(FileConfig):
    """
    :param out_dir: output directory, ignored if init_from is not 'resume'
    :param checkpoint_name: name of the checkpoint to load, ignored if init_from is not 'resume'
    :param init_from: either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    :param seed: random seed
    :param torch_compile: use PyTorch 2.0 to compile the model to be faster
    :param device: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    """
    out_dir: Path = ''
    checkpoint_name: str = 'last'
    init_from: str = 'resume'
    seed: int = 1337
    torch_compile: bool = False
    device: str = None

    def __post_init__(self):
        super().__post_init__()

        if self.device is None:
            self.device = get_default_device()


@dataclass
class TextConfig(DataclassUtils):
    """
    :param start: prompt to start generation.
      Can be "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    :param num_samples: number of samples to draw
    :param max_new_tokens: number of tokens generated in each sample
    :param temperature: 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    :param top_k: retain only the top_k most likely tokens, clamp others to have 0 probability
    """
    start: str = "\n"
    num_samples: int = 1
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 200

    def __post_init__(self):
        if self.start.startswith('FILE:'):
            with open(self.start[5:], 'r', encoding='utf-8') as f:
                self.start = f.read()


@dataclass
class FullConfig(TextConfig, ModelConfig):
    def __post_init__(self):
        TextConfig.__post_init__(self)
        ModelConfig.__post_init__(self)


class Sampler:
    def __init__(self, **kwargs):
        self.config = config = ModelConfig(**kwargs)

        print(f'Using device {config.device}')

        # 'float32' or 'bfloat16' or 'float16'
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        device_type = 'cuda' if 'cuda' in config.device else 'cpu'  # for later use in torch.autocast
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        checkpoint = None
        if config.init_from == 'resume':
            # init from a model saved in a specific directory
            out_dir = DIR / 'results' / config.out_dir
            model_loader = ModelLoader(
                out_dir=out_dir,
                device=config.device,
                checkpoint_name=config.checkpoint_name,
            )
            self.model = model_loader.load()
            checkpoint = model_loader.checkpoint
        elif config.init_from.startswith('gpt2'):
            # init from a given GPT-2 model
            self.model = GPT.from_pretrained(config.init_from, dict(dropout=0.0))
        else:
            raise ValueError(f"Unknown init_from {config.init_from}, must be 'resume' or one type of 'gpt2'")

        self.model.eval()
        self.model.to(config.device)
        if config.torch_compile:
            self.model = torch.compile(self.model)  # requires PyTorch 2.0 (optional)

        # look for the meta pickle in case it is available in the dataset folder
        meta_path = False
        if config.init_from == 'resume' and 'config' in checkpoint:
            _ckpt_config = checkpoint['config']
            if 'dataset' in _ckpt_config:  # older results might not have these...
                dataset = _ckpt_config['dataset']
                if os.path.exists(_ := DIR / 'data' / dataset / 'meta.pkl'):
                    meta_path = _

        if meta_path:
            print(f"Loading meta from {meta_path}...")
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            self.encode = lambda s: [stoi[c] for c in s]
            self.decode = lambda l: ''.join([itos[i] for i in l])
        else:
            # assume gpt-2 encodings by default
            print("No meta.pkl found, assuming GPT-2 encodings")
            enc = tiktoken.get_encoding("gpt2")
            self.encode = partial(enc.encode, allowed_special={"<|endoftext|>"})
            self.decode = enc.decode

    def generate_text(self, **kwargs):
        config = TextConfig(**kwargs)
        start_ids = self.encode(config.start)
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])

        result = []
        with torch.no_grad(), self.ctx:
            for k in range(config.num_samples):
                y = self.model.generate(
                    idx=x,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_k=config.top_k
                )
                result.append(self.decode(y[0].tolist()))

        return unbox(result)

    @property
    def device(self):
        return self.config.device


def generate_text(**kwargs):
    """
    Sample from a trained model
    """
    config = FullConfig(**kwargs)
    model_config = get_relevant_config(subclass=ModelConfig, config=config)
    model_config.pop('config_file', None)
    sampler = Sampler(**model_config)

    text_config = get_relevant_config(subclass=TextConfig, config=config)
    return sampler.generate_text(**text_config)
