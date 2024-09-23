import math
import os
import pickle
import time
from contextlib import nullcontext, contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from llm.cache.disk import get_disk
from llm.configurator import FileConfig
from llm.loader import load_data
from llm.model import GPTConfig, GPT
from llm.utils import ModelLoader, get_default_device

DIR = Path(__file__).parent


@dataclass
class TrainingConfig(FileConfig):
    """
    :param out_dir: output directory, ignored if init_from is not 'resume'
    :param init_from: either 'resume' (from an out_dir), 'scratch', or a gpt2 variant (e.g. 'gpt2-xl')
    :param seed: random seed
    :param torch_compile: use PyTorch 2.0 to compile the model to be faster
    :param device: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    :param dtype: 'float16', 'bfloat16', 'float32', 'float64' etc.
    :param train_ratio: what fraction of the data to use for training
    :param eval_interval: how often to evaluate the model
    :param log_interval: how often to log loss info
    :param eval_iters: how many iters to evaluate
    :param eval_only: if True, run eval only
    :param always_save_checkpoint: if True, save checkpoints after each eval
    :param override_checkpoint: if True, override the previous checkpoint
    :param wandb_log: if True, log with wandb
    :param wandb_project: wandb project name
    :param wandb_run_name: wandb run name
    :param dataset: dataset name
    :param gradient_accumulation_steps: gradient accumulation steps, used to simulate larger batch sizes
    :param batch_size: batch size. if gradient_accumulation_steps > 1, this is the micro-batch size
    :param block_size: block size
    :param n_layer: number of layers
    :param n_head: number of heads
    :param n_embd: embedding dimension
    :param dropout: dropout. for pretraining 0 is good, for finetuning try 0.1+
    :param bias: bias
    :param learning_rate: learning rate
    :param max_iters: maximum number of iterations
    :param weight_decay: weight decay
    :param beta1: beta1 for adam
    :param beta2: beta2 for adam
    :param grad_clip: gradient clipping. disable if == 0.0
    :param decay_lr: whether to decay learning rate
    :param warmup_iters: warmup iterations
    :param lr_decay_iters: learning rate decay iterations. should be ~= max_iters per Chinchilla
    :param min_lr: minimum learning rate. should be ~= learning_rate/10 per Chinchilla
    :param backend: DDP backend. 'nccl', 'gloo', etc.
    :param patience: how many unimproved evals to wait before early stopping
    """
    out_dir: Path = 'out'
    training_data_path: str = None
    init_from: str = 'scratch'
    seed: int = 1337
    torch_compile: bool = True
    device: str = None
    dtype: str = None
    train_ratio: float = 0.9
    log_interval: int = 1
    eval_interval: int = 100
    eval_iters: int = 20
    eval_only: bool = False
    always_save_checkpoint: bool = False
    override_checkpoint: bool = True
    wandb_log: bool = False
    wandb_project: str = 'owt'
    wandb_run_name: str = 'gpt2'
    encoding: str = 'gpt2'
    dataset: str = 'openwebtext'
    gradient_accumulation_steps: int = 5 * 8
    batch_size: int = 12
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5
    backend: str = 'nccl'
    patience: int = 5

    def __post_init__(self):
        super().__post_init__()

        if self.device is None:
            self.device = get_default_device()
        print(f'Using device {self.device}')

        if self.dtype is None:
            # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
            self.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

        if not isinstance(self.out_dir, Path):
            self.out_dir = Path(self.out_dir)
        if not self.out_dir.is_absolute():
            self.out_dir = DIR / 'results' / self.out_dir


class Trainer:
    """
    This trainer can be run both on a single gpu in debug mode,
    and also in a larger training run with distributed data parallel (ddp).

    To run on a single GPU, example:
    $ python train.py

    To run with DDP on 4 gpus on 1 node, example:
    $ torchrun --standalone --nproc_per_node=4 train.py

    To run with DDP on 4 gpus across 2 nodes, example:
    - Run on the first (master) node with example IP 123.456.123.456:
    $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
    - Run on the worker node:
    $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
    (If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
    """

    def __init__(self, **kwargs):
        self.config = TrainingConfig(**kwargs)

        # if not ddp, we are running on a single gpu, and one process
        self.master_process = True
        self.seed_offset = 0
        self.ctx = nullcontext()
        self.model = None
        self.data_dir = None

        self.ddp_world_size = 1
        self.ddp_local_rank = None
        self.ddp = int(
            os.environ.get('RANK', -1)) != -1  # is this a distributed data parallel run (multiple GPU nodes)?

    @property
    def device(self):
        return self.config.device

    @device.setter
    def device(self, value):
        self.config.device = value

    @property
    def is_cuda(self):
        return self.config.device.startswith('cuda')

    @property
    def device_type(self):
        return 'cuda' if self.is_cuda else 'cpu'

    def _setup_ddp(self):
        if self.ddp:
            init_process_group(backend=self.config.backend)
            ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])  # ID (within the local node / server) of the GPU to use
            ddp_world_size = int(os.environ['WORLD_SIZE'])  # number of processes / GPUs in DDP
            self.device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(self.device)
            self.master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
            self.seed_offset = ddp_rank  # each process gets a different seed
            # world_size number of processes will be training simultaneously, so we can scale
            # down the desired gradient accumulation iterations per process proportionally
            assert self.config.gradient_accumulation_steps % ddp_world_size == 0
            self.config.gradient_accumulation_steps //= ddp_world_size
        tokens_per_iter = self.config.gradient_accumulation_steps * self.ddp_world_size * self.config.batch_size * self.config.block_size
        print(f"tokens per iteration will be: {tokens_per_iter:,}")

    def _get_batch(self, split: str):
        """
        Generate a small batch of data of inputs x and targets y

        :param split: train or val
        :return: (x, y)
        """
        config = self.config

        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        # data = np.memmap(cache[f"input_ids_{dataset}_{split}"], dtype=np.uint16, mode='r')
        data = np.memmap(self.data_dir / f'{split}.bin', dtype=np.uint16, mode='r')

        ix = torch.randint(len(data) - config.block_size, (config.batch_size,))

        x = torch.stack([torch.from_numpy((data[i:i + config.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + config.block_size]).astype(np.int64)) for i in ix])

        if self.is_cuda:
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            _f = lambda _: _.pin_memory().to(self.device, non_blocking=True)
        else:
            _f = lambda _: _.to(self.device)

        x, y = map(_f, [x, y])

        return x, y

    @torch.no_grad()
    def estimate_loss(self):
        """
        Helps estimate an arbitrarily accurate loss over either split using many batches
        """
        results = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self._get_batch(split)
                with self.ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            results[split] = losses.mean()
        self.model.train()
        return results

    def _get_lr(self, it):
        """
        Learning rate decay scheduler (cosine with warmup)
        """
        config = self.config

        if not config.decay_lr:
            return config.learning_rate

        # 1) linear warmup for warmup_iters steps
        if it < config.warmup_iters:
            return config.learning_rate * it / config.warmup_iters

        # 2) if it > lr_decay_iters, return min learning rate
        if it > config.lr_decay_iters:
            return config.min_lr

        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)

    @contextmanager
    def _ddp_ctx(self):
        """
        Wrap model into DDP container
        """
        if self.ddp:
            try:
                self.model = DDP(self.model, device_ids=[self.ddp_local_rank])
                yield
            finally:
                destroy_process_group()
        else:
            yield

    def run(self):
        config = self.config

        self._setup_ddp()

        if self.master_process:
            os.makedirs(config.out_dir, exist_ok=True)

        torch.manual_seed(config.seed + self.seed_offset)

        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        if self.is_cuda:
            # note: float16 data type will automatically use a GradScaler
            pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
            self.ctx = torch.amp.autocast(device_type='cuda', dtype=pt_dtype)

        self.data_dir = load_data(
            name=config.dataset,
            train_ratio=config.train_ratio,
            data_path=config.training_data_path,
            encoding=config.encoding,
            out_dir=config.out_dir,
        )

        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        start_iter_num = 0
        best_val_loss = 1e9

        # attempt to derive vocab_size from the dataset
        meta_vocab_size = None
        meta_path = config.out_dir / 'meta.pkl'
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta.get('stoi'), meta.get('itos')
            if stoi is not None and itos is not None:
                assert len(stoi) == len(itos), f"{len(stoi)} != {len(itos)}"
                meta_vocab_size = len(stoi)
                print(f"found vocab_size = {meta_vocab_size}")

        # When to save the init model (cannot save it just after initialization)
        iter_save_init_ckpt = 0 if config.init_from.startswith('gpt2') else min(config.eval_interval, 10)

        # model init
        checkpoint = None
        model_args = dict(
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            block_size=config.block_size,
            bias=config.bias,
            vocab_size=0,
            dropout=config.dropout,
        )  # start with model_args from command line
        if config.init_from == 'scratch':
            # init a new model from scratch
            print("Initializing a new model from scratch")
            # determine the vocab size we'll use for from-scratch training
            if meta_vocab_size is not None:
                model_args['vocab_size'] = meta_vocab_size
            else:
                print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
                model_args['vocab_size'] = 50304
            gpt_conf = GPTConfig(**model_args)
            self.model = GPT(gpt_conf)
        elif config.init_from == 'resume':
            print(f"Resuming training from {config.out_dir}")
            # resume training from a checkpoint.
            model_loader = ModelLoader(
                out_dir=config.out_dir,
                device=self.device,
                dropout=config.dropout,
            )
            self.model = model_loader.load()
            checkpoint = model_loader.checkpoint

            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            model_args = model_loader.checkpoint['model_args']

            start_iter_num = model_loader.checkpoint['iter_num']
            best_val_loss = model_loader.checkpoint['best_val_loss']
        elif config.init_from.startswith('gpt2'):
            print(f"Initializing from OpenAI GPT-2 weights: {config.init_from}")
            # initialize from OpenAI GPT-2 weights
            override_args = dict(dropout=config.dropout)
            self.model = GPT.from_pretrained(config.init_from, override_args)
            # read off the created config params, so we can store them into checkpoint correctly
            for k in model_args.keys():
                if k not in override_args:
                    model_args[k] = getattr(self.model.config, k)
        else:
            raise ValueError(f"unknown init_from option: {config.init_from}")

        # crop down the model block size if desired, using model surgery
        if config.block_size < self.model.config.block_size:
            self.model.crop_block_size(config.block_size)
            model_args['block_size'] = config.block_size  # so that the checkpoint will have the right value

        self.model.to(self.device)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.amp.GradScaler(device=self.device, enabled=(config.dtype == 'float16'))

        # optimizer
        optimizer = self.model.configure_optimizers(
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            betas=(config.beta1, config.beta2),
            device_type=self.device_type,
        )

        if config.init_from == 'resume':
            optimizer.load_state_dict(checkpoint['optimizer'])
        checkpoint = None  # free up memory

        # compile the model
        if config.torch_compile:
            print("compiling the model...")
            self.model = torch.compile(self.model)  # requires PyTorch 2.0

        with self._ddp_ctx():

            # logging
            if config.wandb_log and self.master_process:
                wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=config.dict())

            # training loop
            X, Y = self._get_batch('train')  # fetch the very first batch
            t0 = time.time()
            local_iter_num = 0  # number of iterations in the lifetime of this process
            raw_model = self.model.module if self.ddp else self.model  # unwrap DDP container if needed
            running_mfu = None
            val_loss_not_improved_count = 0
            for iter_num in range(start_iter_num, config.max_iters):
                lr = self._get_lr(iter_num)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # evaluate the loss on train/val sets and write results
                if (iter_num % config.eval_interval == 0 or iter_num == iter_save_init_ckpt) and self.master_process:
                    losses = self.estimate_loss()
                    print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                    if config.wandb_log:
                        wandb.log({
                            "iter": iter_num,
                            "train/loss": losses['train'],
                            "val/loss": losses['val'],
                            "lr": lr,
                            "mfu": running_mfu * 100 if running_mfu is not None else None,  # convert to percentage
                        })
                    val_loss_improved = losses['val'] < best_val_loss
                    if val_loss_improved:
                        val_loss_not_improved_count = 0
                    else:
                        val_loss_not_improved_count += 1
                    if (val_loss_improved or config.always_save_checkpoint) and iter_num > 0:
                        best_val_loss = losses['val']
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': config.dict(),
                        }
                        if iter_num == iter_save_init_ckpt:
                            filename = 'ckpt_init.pt'
                        elif config.override_checkpoint:
                            filename = f'ckpt.pt'
                        else:
                            filename = f'ckpt_{iter_num:06}.pt'
                        file_path = config.out_dir / filename
                        print(f"saving checkpoint to {file_path}")
                        torch.save(checkpoint, file_path)

                if val_loss_not_improved_count > config.patience:
                    print(f"Validation loss did not improve for {config.patience} iterations, stopping now "
                          f"as it is unlikely to get better.")
                    break

                if iter_num == 0 and config.eval_only:
                    break

                # forward backward update, with optional gradient accumulation to simulate larger batch size
                # and using the GradScaler if data type is float16
                for micro_step in range(config.gradient_accumulation_steps):
                    if self.ddp:
                        # in DDP training we only need to sync gradients at the last micro step.
                        # the official way to do this is with model.no_sync() context manager, but
                        # I really dislike that this bloats the code and forces us to repeat code
                        # looking at the source of that context manager, it just toggles this variable
                        self.model.require_backward_grad_sync = (micro_step == config.gradient_accumulation_steps - 1)
                    with self.ctx:
                        logits, loss = self.model(X, Y)
                        loss /= config.gradient_accumulation_steps  # scale the loss to account for gradient accumulation
                    # immediately async prefetch next batch while model is doing the forward pass on the GPU
                    X, Y = self._get_batch('train')
                    # backward pass, with gradient scaling if training in fp16
                    scaler.scale(loss).backward()
                # clip the gradient
                if config.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip)
                # step the optimizer and scaler if training in fp16
                scaler.step(optimizer)
                scaler.update()
                # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)

                # timing and logging
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                if iter_num % config.log_interval == 0 and self.master_process:
                    # get loss as float. note: this is a CPU-GPU sync point
                    # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                    lossf = loss.item() * config.gradient_accumulation_steps
                    message = f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms"
                    if local_iter_num >= 5 and self.is_cuda:  # let the training loop settle a bit
                        mfu = raw_model.estimate_mfu(config.batch_size * config.gradient_accumulation_steps, dt)
                        if mfu is not None:
                            running_mfu = mfu if running_mfu is None else 0.9 * running_mfu + 0.1 * mfu
                            message += f", mfu {running_mfu * 100:.2f}%"
                    print(message)
                local_iter_num += 1


if __name__ == '__main__':
    config_file = DIR / 'config' / 'train.json'
    trainer = Trainer(config_file=config_file)
    trainer.run()
