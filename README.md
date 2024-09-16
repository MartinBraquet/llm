# Large Language Models

Some experiments with large language models from scratch (i.e., without using any external resources such as the OpenAI API).

## Demo

### Learning a book from scratch

One use-case is to learn a book from scratch. That is, we do not pull weights or any other information than the text in the book itself.

Here we train a model on "The Little Prince" by Antoine de Saint-Exupery.
- 28M parameters
- Size: 350 MB
- GPT-2 encoding
- 6 layers
- 6 attention heads per layer
- Training: on 16-vCPU GPU with 20GB RAM
  - Time: 5 min
  - Cost: $0.2 on runpod
  - We let it overfit ("learning" the book)

Once trained, we can use it to generate text. Here is an example for the input "And now here is my secret":

```shell
python sample.py --out_dir=out-prince --start="And now here is my secret"
```
Output:
```text
And now here is my secret, a very simple secret: It is only with the 
heart that one can see rightly; what is essential is invisible to the eye.” 

“What is essential is invisible to the eye,” the little prince repeated, so that he would be sure to 
remember. 

“It is the time you have wasted for your rose that makes your rose so important.” 

“It is the time I have wasted for my rose...” said the little prince, so that he would be sure to 
remember. 

“Men have forgotten this truth,” said the fox. “But you must not forget it. You become 
responsible, forever, for what you have tamed. You are responsible for your rose...” 

“I am responsible for my rose,” the little prince repeated, so that he would be sure to remember. 
```
The model, strongly overfitting, outputs what follows "And now here is my secret" in the book.
In general, it tends to generate a subset of the book (the most appropriate based on the input).

It did not build any skill to summarize, recognize patterns, nor understand different writing styles. 
Indeed, prompting "Summarize The Little Prince" miserably outputs:

```text
Summarize The Little Prince in matters of death. I owned a silk scarf,” he said, “I could put it around my 
neck and take it away with me. If I owned a flower, I could pluck that flower and take it away with 
me. But you cannot pluck the stars from heaven...” 
```

This model is, however, very small and hence very fast to train. This makes it convenient for applications where
one would like to complete snippets from a book.

### Learning a book by fine-tuning a pretrained model

TODO

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
python data/prince/prepare.py
python train.py config/train_prince.py
python sample.py --out_dir=out-prince
python sample.py --out_dir=out-prince --start="Then the little prince said"
python sample.py --out_dir=out-prince --checkpoint_name=ckpt_init.pt
```

## Developer Notes

I could not run it on an AMD GPU with `torch_directml`
because many operations, such as `torch._foreach_add_`, are not supported by this package (as of `0.2.4.dev240815`).
ROCm might make it work though.
