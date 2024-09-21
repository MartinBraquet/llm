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
| Before finetuning | _The essential is invisible to the common man, but that which is essential to us so is called in the proper name. And, therefore, the power or work of generation is an essential to the nature of an orderly creature, for it serves for the preservation of the natural order of the generation.<br/> Secondly, God by the very power which is ordained to man, produces a true and sensible image of himself, as has been observed._                                                                                                                                                            |
| After finetuning  | _The essential is invisible to the eye; the indispensable is invisible to the mind. It is what makes one day different from other days, one hour from other hours. It is what makes one day a good day, and one hour a bad day.<br/> So then I chose a flower that was unique in the world. I chose the rose that belonged to me. And it is she that I shall put under the glass globe. To protect her, I shall place her under the shelter of my glass globe._ |

We notice a clear change of style and ideas, although the model sometimes fails to reflect the ideology of the book (as attested in "the indispensable is invisible to the mind").

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

### Conclusion

There is nice progress compared to training a model from scratch, but it does not seem to be able to summarize the book upon mention of the title. 
Indeed, since GPT-2 XL can't even do "one plus one" (try it [here](https://huggingface.co/openai-community/gpt2-xl?text=One+plus+one+equals)), it would be unreasonable to expect such a task from it.
Larger models like GPT-3+ can do so.

Disclaimer: the above examples have been cherry-picked to show the best results that such LLM can achieve.
Do not build any type of scientific induction or conclusion from those examples, or you will be commiting an infamous selection bias.

## Installation

### Prerequisites

If running on a Linux machine without GPU, run this beforehand:
```shell
pip install torch --index-url https://download.pytorch.org/whl/cpu
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
torchrun --standalone --nproc_per_node=2 train.py config/train_gpt2_finetuned.json --init_from=resume
python sample.py --out_dir=out-prince
python sample.py --out_dir=out-prince --start="Then the little prince said"
python sample.py --out_dir=out-prince --checkpoint_name=ckpt_init.pt
```

## Tests

```shell
pytest
```

## Developer Notes

I could not run it on an AMD GPU with `torch_directml`
because many operations, such as `torch._foreach_add_`, are not supported by this package (as of `0.2.4.dev240815`).
ROCm might make it work though.
