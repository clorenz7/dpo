from unittest import TestCase

from datasets import load_dataset

from cdpo import model_ops
from cdpo.data_utils import get_default_dir


class TestModelBase(TestCase):

    @classmethod
    def setUpClass(cls):

        cls.model, cls.tokenizer = model_ops.get_partially_trainable_model(
            "openai-community/gpt2", n_layers_freeze=0, dropout=0.02
        )
        cls.model.to("cuda:0")

        ds = load_dataset("Anthropic/hh-rlhf")
        ds['train'] = ds['train'].select(range(128)).filter(lambda x: len(x['chosen']) <= 1024)
        cls.ds = ds

        cls.BASE_DIR = get_default_dir()
