# train a miniature gpt-2 encoded prince model
init_from = 'resume'
out_dir = 'prince_xs'
eval_interval = 10 # keep frequent because we'll overfit
eval_iters = 50 # number of iterations to do during eval
log_interval = 1

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'prince'
wandb_run_name = 'mini-gpt'

dataset = 'prince'
gradient_accumulation_steps = 32
batch_size = 4
block_size = 8

# baby GPT model
n_layer = 1
n_head = 1
n_embd = 4
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
