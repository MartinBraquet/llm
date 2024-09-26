import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch

from llm.model import GPTConfig, GPT

DIR = Path(__file__).parent


def get_last_checkpoint(model_path):
    """
    Init from the last checkpoint in the model_path
    """
    ckpt_list = [f for f in os.listdir(model_path) if 'ckpt' in f]
    if not ckpt_list:
        raise ValueError(f'no checkpoints found in {model_path}')
    if 'ckpt.pt' in ckpt_list:
        return 'ckpt.pt'
    ckpt_list = sorted(ckpt_list)
    checkpoint_name = ckpt_list[-1]
    if checkpoint_name == 'ckpt_init.pt' and len(ckpt_list) > 1:
        checkpoint_name = ckpt_list[-2]
    return checkpoint_name


def parse_model_path(model_path):
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    # if not model_path.is_absolute():
    #     model_path = DIR / 'results' / model_path
    return model_path


class ModelLoader:
    def __init__(
        self,
        model_path: Path | str,
        checkpoint_name: str = 'last',
        device: str = 'cuda',
        dropout: Optional[float] = None
    ):
        model_path = parse_model_path(model_path)
        if checkpoint_name == 'last':
            checkpoint_name = get_last_checkpoint(model_path)
        self.checkpoint_name = checkpoint_name
        self.ckpt_path = model_path / checkpoint_name
        print(f'Using model in {self.ckpt_path}')
        if device.startswith('cuda') and not torch.cuda.is_available():
            device = 'cpu'
        self.device = device
        self.dropout = dropout

        self._model = None
        self._checkpoint = None

    @property
    def model(self):
        return self._model

    @property
    def checkpoint(self):
        return self._checkpoint

    def load(self) -> GPT:
        if self._model is None:
            self._checkpoint = torch.load(
                self.ckpt_path,
                map_location=self.device,
                weights_only=False
            )
            checkpoint_model_args = self._checkpoint['model_args']
            if self.dropout is not None:
                checkpoint_model_args['dropout'] = self.dropout
            gpt_conf = GPTConfig(**checkpoint_model_args)
            self._model = GPT(gpt_conf)

            # fix the keys of the state dictionary :(
            # honestly no idea how results sometimes get this prefix, have to debug more
            state_dict = self._checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

            self._model.load_state_dict(state_dict)

        return self._model


def unbox(e):
    """
    Returns the only element of e if it has only one element, otherwise returns e
    """
    if isinstance(e, str):
        return e
    if isinstance(e, dict):
        e = e.values()
    return next(iter(e)) if hasattr(e, '__len__') and len(e) == 1 else e


def box(e):
    """
    Box a single element into a list
    :param e:
    :return:
    """
    if isinstance(e, (list, tuple, set)):
        return e
    return [e]


@dataclass
class DataclassUtils:

    @classmethod
    def keys(cls):
        return list(cls.__dataclass_fields__)

    def dict(self):
        return {k: getattr(self, k) for k in self.keys()}


@lru_cache
def get_default_device():
    # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def str_to_hash(s):
    return hashlib.sha256(s.encode()).hexdigest()


def list_to_hash(items):
    return str_to_hash('::'.join(items))


def to_path(s):
    if isinstance(s, str):
        return Path(s)
    return s


def make_dir(name):
    os.makedirs(name, exist_ok=True)


class Missing:
    def __repr__(self):
        return 'MISSING'


MISSING = Missing()
