# Large Language Models

[![CI](https://github.com/MartinBraquet/llm/actions/workflows/ci.yml/badge.svg)](https://github.com/MartinBraquet/llm/actions/workflows/ci.yml)
[![CD](https://github.com/MartinBraquet/llm/actions/workflows/cd.yml/badge.svg)](https://github.com/MartinBraquet/llm/actions/workflows/cd.yml)
[![Coverage](https://codecov.io/gh/MartinBraquet/llm/branch/main/graph/badge.svg)](https://codecov.io/gh/MartinBraquet/llm)

Some experiments with large language models from scratch (i.e., without using any external resources such as the OpenAI
API).

## Demo

### Learning a book from scratch

One use-case is to learn a book from scratch. That is, we do not pull weights or any other information than the text in
the book itself.

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

<blockquote>
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
</blockquote>

The model, strongly overfitting, outputs what follows "And now here is my secret" in the book.
In general, it tends to generate a subset of the book (the most appropriate based on the input).

It did not build any skill to summarize, recognize patterns, nor understand different writing styles.
Indeed, prompting "Summarize The Little Prince" miserably outputs:

<blockquote>
Summarize The Little Prince in matters of death. I owned a silk scarf,” he said, “I could put it around my 
neck and take it away with me. If I owned a flower, I could pluck that flower and take it away with 
me. But you cannot pluck the stars from heaven...” 
</blockquote>

This model is, however, very small and hence very fast to train. This makes it convenient for applications where
one would like to complete snippets from a book.

To reproduce the results, visit the [TODO]() notebook.

### Learning a book by fine-tuning a pretrained model

Another use-case is to learn a book by fine-tuning a pretrained model.
That is, we create a model with the same configuration as GPT-2, we pull the weights from GPT-2,
and we slightly update those weights by training the model on the text in the book.

- 1.5B parameters
- Size: 6 GB
- GPT-2 encoding
- 48 layers
- 25 attention heads per layer
- 1600 embedding dimensions
- Training: on 24-vCPU GPU with 80GB RAM
    - Time: 5 min
    - Cost: $0.5 on runpod
    - We let it slightly overfit

We compare the results before and after fine-tuning a GPT2 model on The Little Prince:

| Input             | _The essential is invisible to_                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Before finetuning | _The essential is invisible to the common man, but that which is essential to us so is called in the proper name. And, therefore, the power or work of generation is an essential to the nature of an orderly creature, for it serves for the preservation of the natural order of the generation.<br/> Secondly, God by the very power which is ordained to man, produces a true and sensible image of himself, as has been observed._                         |
| After finetuning  | _The essential is invisible to the eye; the indispensable is invisible to the mind. It is what makes one day different from other days, one hour from other hours. It is what makes one day a good day, and one hour a bad day.<br/> So then I chose a flower that was unique in the world. I chose the rose that belonged to me. And it is she that I shall put under the glass globe. To protect her, I shall place her under the shelter of my glass globe._ |

We notice a clear change of style and ideas, although the model sometimes fails to reflect the ideology of the book (as
attested in "the indispensable is invisible to the mind").

Output without input prompt (i.e., `\n`):
<blockquote>
This time, the little prince came back to his idea. 

“I myself own a flower. Do you think that her colour is an accident of birth?”

“Of course it is.” The businessman raised his head. “Flowers have been growing thorns
for a long time. And if the thorns are not stopped, they bring disaster on the plants.”

“Then my flower is obviously a very dangerous flower...” “The thorns are of no use at all. The plant is
alike fragile and vulnerable. One must not destroy it but rather educate it...”
</blockquote>

Output to "Love is the answer":
<blockquote>
Love is the answer that sometimes requires a little 
meditation. 

I want you to understand, very clearly, why it is that during the fifty-four years that I have had you as my friend,
I have had no desire to harm you. In fact, I have constantly tried to help you. I have tried to
make you happy. I have tried to make you happy when you were angry, and I have tried to make you
happier still when you were happy. Try as I might, I could not make you happy unless you were
reassured.

You see, I do not know how to see sheep from the inside. It is not in my nature. When you were
a little boy, I thought that I was very much like you. I was proud of my hat. I thought that I was unique in all the
world. And you, you were unique in all the world... But you are not at all like me. You are not my son.
You are not my lover. You are not my friend. You are just a little boy who is just like a
hundred thousand other little boys. And I, ah, well... I am not at all proud of that. Not very nearly. But I am
magnificent, nonetheless. Because of you, I have been given a new self-confidence. Because of you, I have
...
boys have been told to do. And that is a great thing! Because of you, I have been loved. Oh, yes. I have!
</blockquote>

The wisdom behind those words, which, for the most part, are not in the book, is quite appreciable.

To reproduce the results, visit the [TODO]() notebook.

### Conclusion

There is nice progress compared to training a model from scratch, but it does not seem to be able to summarize the book
upon mention of the title.
Indeed, since GPT-2 XL can't even do "one plus one" (try
it [here](https://huggingface.co/openai-community/gpt2-xl?text=One+plus+one+equals)), it would be unreasonable to expect
such a task from it.
Larger models like GPT-3+ can do so.

Disclaimer: the above examples have been cherry-picked to show the best results that such LLM can achieve.
Do not build any type of scientific induction or conclusion from those examples, or you will be commiting an infamous
selection bias.

## Installation

```shell
git clone git@github.com:MartinBraquet/llm.git
cd llm
```

### Environment

If not already done, create a virtual environment using your favorite package manager. For instance using conda:

```shell
conda create -n llm python=3.11
conda activate llm
```

### Prerequisites

If running on a Linux machine without intent to use a GPU, run this beforehand:

```shell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Main Installation

```shell
pip install -e "."
```

## Usage

This package can be used in the following ways:

### Training

One can train a model from scratch via:

```python
from llm.train import Trainer

trainer = Trainer(
    out_dir='kafka',  # output directory where the model will be saved
    dataset='https://www.gutenberg.org/cache/epub/5200/pg5200.txt',  # dataset URL or local path
    eval_interval=10,  # when to evaluate the model
    batch_size=4,  # batch size
    block_size=8,  # block size (aka context length)
    n_layer=1,  # number of layers
    n_head=1,  # number of attention heads per layer
    n_embd=4,  # embedding dimension
    dropout=0.2,  # dropout rate
    learning_rate=0.01,  # learning rate
    beta2=0.99,  # adam beta2 (should be reduced for larger models / datasets)
)
trainer.run()
```

It should take a few minutes to train on a CPU, and it is much faster on a GPU.

Note that there are many more parameters to tweak, if desired. See all of them in the doc:

```python
from llm.train import Trainer

Trainer.help_config()
```

It will stop training when the evaluation loss stops improving. Once done, one can generate text from it; see the next
section below (setting the appropriate value for `out_dir`, e.g., `'kafka'`).

### Text Generation

One can generate text from a trained model via:

```python
from llm.sample import Sampler

sampler = Sampler(
    out_dir='kafka',  # output directory where the model has been saved
    seed=None,
    # set the seed to any integer to remove randomness (i.e., always produce the same output for the same input)
)
generated_text = sampler.generate_text(
    prompt='My name is Martin and I am',  # prompt
    max_tokens=100,  # number of tokens to generate
)
print(generated_text)
```

To access to all the parameters for text generation, see the doc:

```python
from llm.sample import Sampler

Sampler.help_model_config()  # for the arguments to Sampler
Sampler.help_text_config()  # for the arguments to Sampler.generate_text
```

#### From a pre-trained model

If you do not want to train a model, as described in the [Training](#Training) section, you can still generate text from
a pre-trained model available online. After passing `init_from='online'`, you can set `out_dir` to any of those currently supported models:

| `out_dir`     | # layers | # heads | embed dims | # params | size   |
|---------------|----------|---------|------------|----------|--------|
| `gpt2`        | 12       | 12      | 768        | 124M     | 500 MB |
| `gpt2-medium` | 24       | 16      | 1024       | 350M     | 1.4 GB |
| `gpt2-large`  | 36       | 20      | 1280       | 774M     | 3 GB   |
| `gpt2-xl`     | 48       | 25      | 1600       | 1558M    | 6 GB   |

Note that the first time you use a model, it needs to be downloaded from the internet; so it can take a few minutes.

Example:

```python
from llm.sample import Sampler

sampler = Sampler(init_from='online', out_dir='gpt2')
print(sampler.generate_text(prompt='Today I decided to'))
```

## Tests

```shell
pytest llm
```

## Feedback

For any issue / bug report / feature request,
open an [issue](https://github.com/MartinBraquet/llm/issues).

## Contributions

To provide upgrades or fixes, open a [pull request](https://github.com/MartinBraquet/llm/pulls).

### Contributors

[![Contributors](https://contrib.rocks/image?repo=MartinBraquet/llm)](https://github.com/MartinBraquet/llm/graphs/contributors)

## Developer Notes

I could not run it on an AMD GPU with `torch_directml`
because many operations, such as `torch._foreach_add_`, are not supported by this package (as of `0.2.4.dev240815`).
ROCm might make it work though.
