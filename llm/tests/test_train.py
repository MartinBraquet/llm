from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from llm.train import Trainer

DIR = Path(__file__).parent


class FakeWandb:
    @classmethod
    def init(cls, **kwargs):
        pass

    @classmethod
    def log(cls, **kwargs):
        pass


class TestTrain(TestCase):

    def test_train(self):
        config = dict(
            init_from="scratch",
            model_path=DIR / 'results' / ".out_test_train",
            training_data_path="prince",
            torch_compile=False,
            log_interval=9,
            eval_interval=2,
            eval_iters=2,
            batch_size=1,
            block_size=4,
            n_layer=2,
            n_head=2,
            n_embd=4,
            dropout=0.2,
            learning_rate=10e-3,
            max_iters=20,
            lr_decay_iters=20,
            min_lr=5e-3,
            beta2=0.99,
            patience=2
        )
        trainer = Trainer(**config)
        trainer.run()

        self.assertFalse(trainer.resume())

    def test_train_resume(self):
        trainer = Trainer(
            init_from="resume",
            model_path=DIR / 'results' / "test1",
            training_data_path="prince",
            max_iters=20,
        )
        trainer.run()

    @patch('wandb.init', new=FakeWandb.init)
    @patch('wandb.log', new=FakeWandb.log)
    def test_train_wandb(self):
        trainer = Trainer(
            init_from="resume",
            model_path=DIR / 'results' / "test1",
            training_data_path="prince",
            max_iters=20,
            wandb_log=True,
        )
        trainer.run()

    def test_train_profile(self):
        trainer = Trainer(
            init_from="resume",
            model_path=DIR / 'results' / "test1",
            training_data_path="prince",
            max_iters=20,
            profile=True,
        )
        trainer.run()

    def test_train_finetune(self):
        trainer = Trainer(
            init_from="gpt2",
            model_path=DIR / 'results' / ".gpt2",
            training_data_path="prince",
            max_iters=2,
        )
        trainer.load_model()
        model = trainer.model
        size = model.get_num_params()
        self.assertEqual(123653376, size)

