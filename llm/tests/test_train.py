from pathlib import Path
from unittest import TestCase

from llm.train import Trainer


DIR = Path(__file__).parent

class TestTrain(TestCase):

    def test_train(self):
        config = {
            "init_from": "scratch",
            "model_path": DIR / 'results' / ".out_test_train",
            "training_data_path": "prince",
            "torch_compile": False,
            "log_interval": 9,
            "eval_interval": 10,
            "eval_iters": 5,
            "batch_size": 1,
            "block_size": 4,
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 4,
            "dropout": 0.2,
            "learning_rate": 10e-3,
            "max_iters": 20,
            "lr_decay_iters": 20,
            "min_lr": 5e-3,
            "beta2": 0.99,
            "warmup_iters": 10,
            "patience": 3
        }
        trainer = Trainer(**config)
        trainer.run()
