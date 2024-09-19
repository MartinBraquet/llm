import os
from pathlib import Path
from typing import Optional

import torch

from llm.model import GPTConfig, GPT


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
        if checkpoint_name == 'last':
            checkpoint_name = get_last_checkpoint(out_dir)
        self.checkpoint_name = checkpoint_name
        print(f'Using model in {checkpoint_name}')
        self.ckpt_path = out_dir / checkpoint_name
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
