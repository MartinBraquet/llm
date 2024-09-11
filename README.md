# Large Language Models

Some experiments with large language models from scratch (i.e., without using any external resources such as the OpenAI API).

Note: I could not run it on an AMD GPU with `torch_directml` (latest version: `0.2.4.dev240815`)
because many operations, such as `torch._foreach_add_`, are not supported by this package.