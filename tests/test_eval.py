
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
