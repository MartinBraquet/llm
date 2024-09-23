from pathlib import Path
from unittest import TestCase

from llm import BASE_DIR
from llm.sample import generate_text, Sampler

DIR = Path(__file__).parent

class TestSample(TestCase):

    def test_sample(self):
        sampler = Sampler(out_dir=BASE_DIR / 'results' / 'test')
        text = sampler.generate_text(start='Love is the answer to', max_new_tokens=20)
        self.assertEqual('Love is the answer to41b1F-.tTwmv-vxL.scT', text)

    # def test_sample_from_file(self):
    #     config_file = DIR / 'config.json'
    #     text = generate_text(config_file=config_file)
    #     self.assertEqual('\natTordyxfu4uAnLj4lch', text)
