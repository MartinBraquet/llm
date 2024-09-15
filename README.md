# Large Language Models

Some experiments with large language models from scratch (i.e., without using any external resources such as the OpenAI API).

Note: I could not run it on an AMD GPU with `torch_directml` (latest version: `0.2.4.dev240815`)
because many operations, such as `torch._foreach_add_`, are not supported by this package.

## Installation

### Prerequisites

If running on a Linux machine without GPU, run this beforehand:
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Main Installation

```shell
pip install -e "."
```

## Usage

```shell
cd llm
python data/dosto/prepare.py
python train.py config/train_dosto.py
python sample.py --out_dir=out-dosto --checkpoint_name=ckpt.pt
```