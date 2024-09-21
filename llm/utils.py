import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch

from llm.model import GPTConfig, GPT

DIR = Path(__file__).parent


def get_last_checkpoint(out_dir):
    """
    Init from the last checkpoint in the out_dir
    """
    ckpt_list = [f for f in os.listdir(out_dir) if 'ckpt' in f]
    if not ckpt_list:
        raise ValueError(f'no checkpoints found in {out_dir}')
    if 'ckpt.pt' in ckpt_list:
        return 'ckpt.pt'
    ckpt_list = sorted(ckpt_list)
    checkpoint_name = ckpt_list[-1]
    if checkpoint_name == 'ckpt_init.pt' and len(ckpt_list) > 1:
        checkpoint_name = ckpt_list[-2]
    return checkpoint_name


class ModelLoader:
    def __init__(
        self,
        out_dir: Path,
        checkpoint_name: str = 'last',
        device: str = 'cuda',
        dropout: Optional[float] = None
    ):
        if not isinstance(out_dir, Path):
            out_dir = Path(out_dir)
        if not out_dir.is_absolute():
            out_dir = DIR / 'results' / out_dir
        if checkpoint_name == 'last':
            checkpoint_name = get_last_checkpoint(out_dir)
        self.checkpoint_name = checkpoint_name
        self.ckpt_path = out_dir / checkpoint_name
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

    def load(self):
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
    print(f'Using device {device}')
    return device
