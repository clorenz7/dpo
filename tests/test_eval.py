
import torch

import cdpo.evaluation

from base import TestModelBase


class TestEval(TestModelBase):

    @torch.no_grad()
    def test_calc_prob(self):
        self.model.eval()
        log_prob = cdpo.evaluation.calc_response_probability(
            self.model, self.tokenizer, self.ds['train'][0]['chosen']
        )

        assert log_prob <= 0

    def test_preproc_dpo(self):

        # Test that we can process a single example
        split_str = "Assistant:"
        example = cdpo.evaluation.preprocess_example_for_dpo(
            self.ds['train'][0], self.model, self.tokenizer,
            split_str=split_str
        )

        # Determine what we expect to get back
        given_text = self.ds['train'][0]['chosen']
        context, response = given_text.rsplit(split_str, 1)
        exp_context = context + split_str
        exp_response = response + self.tokenizer.eos_token

        # Decode the tokens to strings
        idx = example['chosen_start_idx']
        new_context = self.tokenizer.decode(example['chosen'][:idx])
        new_response = self.tokenizer.decode(example['chosen'][idx:])

        assert exp_context == new_context
        assert exp_response == new_response

        # Test that things are their proper type
        assert isinstance(example['chosen'], list)
        assert isinstance(example['rejected'], list)
        assert isinstance(example['chosen_log_prob'], float)

        # Check that it works to run on an entire dataset
        ds_train = cdpo.evaluation.preprocess_dataset_for_dpo(
            self.ds['train'].select(range(8)), self.model, self.tokenizer
        )
        assert len(ds_train) == 8
