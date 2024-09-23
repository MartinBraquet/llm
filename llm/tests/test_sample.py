from pathlib import Path
from unittest import TestCase

from llm import BASE_DIR
from llm.sample import generate_text, Sampler

DIR = Path(__file__).parent

class TestSample(TestCase):

    def test_sample(self):
        sampler = Sampler(out_dir=BASE_DIR / 'results' / 'test_sample')
        text = sampler.generate_text(start='Love is the answer to', max_new_tokens=10)
        self.assertEqual('Love is the answer to you.� also prince little, her\n\n', text)

    def test_sample_from_file(self):
        config_file = DIR / 'config.json'
        text = generate_text(config_file=config_file)
        self.assertEqual('\n Soors from at be And He did much', text)
