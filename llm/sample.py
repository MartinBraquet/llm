import os
import pickle
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import tiktoken
import torch

from llm.configurator import load_config_file
from llm.model import GPT
from llm.utils import ModelLoader, unbox, get_default_device, to_path, parse_model_path, MISSING

DEFAULT_PROMPT = "\n"

DIR = Path(__file__).parent

_DEFAULT_CONFIG = dict(
    device=None,
    _seed=1337,
    torch_compile=False,
    dtype=None,
    model_path='out',
    config_file=None,
)


class ML:
    def __init__(
        self,
        device: str = MISSING,
        seed: int = MISSING,
        torch_compile: bool = MISSING,
        dtype: str = MISSING,
        model_path: Path | str = MISSING,
        config_file: str = MISSING,
        **kwargs
    ):
        """
        :param device: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
        :param seed: random seed. set it to any integer to remove randomness (i.e., always produce the same output for
            the same input)
        :param torch_compile: use PyTorch 2.0 to compile the model to be faster
        :param dtype: 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
        :param model_path: directory to load the model
        :param config_file: config file to load
        """
        self.__dict__.update(_DEFAULT_CONFIG)
        self._process_config_file(config_file)

        if device is not MISSING:
            self.device = device
        elif self.device is None:
            self.device = get_default_device()
        print(f'Using device {self.device}')

        if seed is not MISSING:
            self._seed = seed

        if torch_compile is not MISSING:
            self.torch_compile = torch_compile

        if model_path is not MISSING:
            self.model_path = model_path
        self.model_path = parse_model_path(self.model_path)

        self.model = None

        if dtype is not MISSING:
            self.dtype = dtype
        elif self.dtype is None:
            # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
            self.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        self.ctx = nullcontext()
        if self.is_cuda:
            # note: float16 data type will automatically use a GradScaler
            pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
            self.ctx = torch.amp.autocast(device_type='cuda', dtype=pt_dtype)

    @property
    def model_name(self):
        return Path(self.model_path).stem

    @property
    def is_cuda(self):
        return self.device.startswith('cuda')

    @property
    def device_type(self):
        return 'cuda' if self.is_cuda else 'cpu'

    @property
    def seed(self):
        return self._seed

    def manual_seed(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)

    def compile_model(self):
        if self.torch_compile:
            print("compiling the model...")
            self.model = torch.compile(self.model)  # requires PyTorch 2.0

    def get_config(self):
        """
        Return a dict of the attributes of this class
        """
        skip_attrs = ('model', 'ctx')
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_') and k not in skip_attrs}

    def _process_config_file(self, config_file):
        if config_file is not MISSING:
            config = load_config_file(config_file)
            for k in config.keys():
                if k.startswith('_'):
                    raise ValueError(f"Key {k} cannot start with '_'")
            self.__dict__.update(config)


class Sampler(ML):
    def __init__(
        self,
        checkpoint_name: str = 'last',
        init_from: str = 'resume',
        **kwargs
    ):
        """
        :param checkpoint_name: name of the checkpoint to load, ignored if init_from is not 'resume'
        :param init_from: either 'resume' (from a local model_path) or 'online' (from HuggingFace hub)
        """
        super().__init__(**kwargs)
        self.checkpoint_name = checkpoint_name
        self.init_from = init_from

        self.manual_seed()

        checkpoint = None
        if self.init_from == 'resume':
            # init from a model saved in a specific directory
            model_loader = ModelLoader(
                model_path=self.model_path,
                device=self.device,
                checkpoint_name=self.checkpoint_name,
            )
            self.model = model_loader.load()
            checkpoint = model_loader.checkpoint
        elif self.init_from == 'online':
            # init from a given GPT-2 model
            assert str(self.model_path).startswith('gpt2'), 'model_path must start with gpt2'
            self.model = GPT.from_pretrained(str(self.model_path), dict(dropout=0.0))
        else:
            raise ValueError(f"Unknown init_from {self.init_from}, must be 'resume' or 'online'")

        self.model.eval()
        self.model.to(self.device)
        self.compile_model()

        self._setup_encoding(ckpt_config=checkpoint.get('config') if checkpoint else None)

    def _setup_encoding(self, ckpt_config):
        encoding = None
        if self.init_from == 'resume' and ckpt_config is not None:
            encoding = ckpt_config.get('encoding')
            meta_dir = to_path(ckpt_config.get('data_dir'))
            # if isinstance(meta_dir, Path) and not meta_dir.is_absolute():
            #     meta_dir = BASE_DIR / 'results' / meta_dir
            if meta_dir and os.path.exists(meta_path := meta_dir / 'meta.pkl'):
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                stoi, itos = meta.get('stoi'), meta.get('itos')
                if stoi is not None or itos is not None:
                    print(f"Loading encoding from {meta_path}...")
                    print(f"stoi: {stoi}")
                    print(f"itos: {itos}")
                    self.encode = lambda s: [stoi[c] for c in s]
                    self.decode = lambda l: ''.join([itos[i] for i in l])
                    return

        if encoding in (None, 'gpt2'):
            if encoding is None:
                print("No meta.pkl found, assuming GPT-2 encodings")
            enc = tiktoken.get_encoding("gpt2")
            self.encode = partial(enc.encode, allowed_special={"<|endoftext|>"})
            self.decode = enc.decode
            return

        raise ValueError(f"Unknown encoding {encoding}")

    def generate_text(
        self,
        prompt: str = DEFAULT_PROMPT,
        num_samples: int = 1,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 200,
    ):
        """
        :param prompt: prompt to start generation.
          Can be "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
        :param num_samples: number of samples to draw
        :param max_tokens: number of tokens generated in each sample
        :param temperature: 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        :param top_k: retain only the top_k most likely tokens, clamp others to have 0 probability
        """
        if prompt.startswith('FILE:'):
            with open(prompt[5:], 'r', encoding='utf-8') as f:
                prompt = f.read()

        start_ids = self.encode(prompt)
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])

        result = []
        with torch.no_grad(), self.ctx:
            for k in range(num_samples):
                y = self.model.generate(
                    idx=x,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
                result.append(self.decode(y[0].tolist()))

        return unbox(result)
