import tiktoken

enc = tiktoken.get_encoding("gpt2")

tokens = enc.encode_ordinary("world hello")
print(tokens)

print('"', enc.decode([tokens[0]]), '"')
print('"', enc.decode([tokens[1]]), '"')

print(enc.decode(tokens))

# for i in range(enc.n_vocab):
#     print('"', enc.decode([i]), '"')