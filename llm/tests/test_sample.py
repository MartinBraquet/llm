from pathlib import Path
from unittest import TestCase

from llm.sample import generate_text, Sampler

DIR = Path(__file__).parent

class TestSample(TestCase):

    def test_sample(self):
        sampler = Sampler(out_dir='prince_xs')
        text = sampler.generate_text(start='Love is the answer to', max_new_tokens=10)
        self.assertEqual(text, 'Love is the answer topl answereds from at be And He did understand')

    def test_sample_from_file(self):
        config_file = DIR / 'config.json'
        text = generate_text(config_file=config_file)
        self.assertEqual(text, '\npl answereds from at be And He did understand')
