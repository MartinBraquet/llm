from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from llm import BASE_DIR
from llm.sample import generate_text, Sampler

DIR = Path(__file__).parent


def patch_multinomial_sampling(probs, num_samples):
    """
    Seeded torch.multinomial is platform-dependent, so we can't use it for reproducibility in unit testing
    Using np.random.multinomial allows for reproducibility. But since is less efficient (need to share memory between
    CPU and GPU), we stick to torch.multinomial in production.
    """
    probs = np.array(probs[0]).astype(np.float64)
    probs /= probs.sum()
    idx_next = torch.asarray([[np.argmax(np.random.multinomial(num_samples, probs, size=1)[0])]])
    return idx_next


@patch('torch.multinomial', patch_multinomial_sampling)
class TestSample(TestCase):

    def test_sample(self):
        sampler = Sampler(out_dir=BASE_DIR / 'results' / 'test')
        text = sampler.generate_text(prompt='Love is the answer to', max_tokens=20)
        self.assertEqual('Love is the answer to41b1F-.tTwmv-vxL.scT', text)

    def test_sample_from_file(self):
        config_file = DIR / 'config.json'
        text = generate_text(config_file=config_file)
        self.assertEqual('\n41hF-rgwmv-vxL.scT\nu', text)

    def test_sample_from_online(self):
        sampler = Sampler(init_from='online', out_dir='gpt2')
        text = sampler.generate_text(prompt='The sun', max_tokens=10)
        self.assertEqual('The sun has been shining for weeks, but it has set', text)
