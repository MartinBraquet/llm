import torch

from llm.train import Trainer
from llm.utils import ModelLoader

config_file = 'train_test.json'
trainer = Trainer(config_file=config_file)
trainer.run()

out_dir = 'test'
model_loader = ModelLoader(out_dir=out_dir)
model_loader.load()
config = model_loader.checkpoint['config']
config['out_dir'] = out_dir
config['config_file'] = 'train_test.json'
config['training_data_path'] = 'test/prince.txt'

torch.save(model_loader.checkpoint, model_loader.ckpt_path)
