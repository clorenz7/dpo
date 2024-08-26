# cdpo
Cory's Direct Preference Optimization replication work

The goal is to replicate the dialogue task results (i.e. Figure 3) from the [original DPO paper](https://arxiv.org/abs/2305.18290).

## Using the Repository

Source code can be found in [`src`](./src).

[`scripts/eval_win_rate.py`](./scripts/eval_win_rate.py) can be used to evaluate responses using the ChatGPT AI.

[`scripts/train_and_eval.py`](./scripts/train_and_eval.py) can be used to run the full pipeline of supervised fine tuning (SFT) followed by DPO.

Example configurations files can be found in the [`config`](./config) directory

### Installation

Use an editable install to work with the code locally:

`python -m pip install -e .`

## Results Summary

The performance of supervised fine tuning and direct preference optimization for various model sizes is summarized here:

![final_results](./assets/params_vs_perf_250ex.png)

In order to save time and compute costs, smaller models were used on training and test sets of reduced size. Additional regularization is likely needed on the GPT2-large model. Nevertheless, the ability of DPO to improve chat bot performance has been replicated.


A more thorough write-up of the results can be found [here](./docs/details_Aug2024.md).