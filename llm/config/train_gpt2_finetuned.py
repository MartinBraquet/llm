init_from = 'resume'
out_dir = 'gpt2_finetuned'
eval_interval = 20
eval_iters = 10 # number of iterations to do during eval
log_interval = 2

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True
override_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'gpt2_finetuned'
wandb_run_name = 'mini-gpt'

dataset = 'prince'
gradient_accumulation_steps = 8
batch_size = 4

dropout = 0.2

learning_rate = 3e-4 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = learning_rate / 10 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially
