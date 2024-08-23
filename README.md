# cdpo
Cory's Direct Preference Optimization replication work

The goal is to replicate the dialogue task results (i.e. Figure 3) from the [original DPO paper](https://arxiv.org/abs/2305.18290).

## Using the Repository

Source code can be found in [`src`](./src).

[`scripts/eval_win_rate.py`](./scripts/eval_win_rate.py) can be used to evaluate responses using the ChatGPT AI.

[`scripts/train_and_eval.py`](./scripts/train_and_eval.py) can be used to run the full pipeline of supervised fine tuning (SFT) followed by DPO.

### Installation

Use an editable install to work with the code locally:

`python -m pip install -e .`

## Results

The performance of supervised fine tuning and direct preference optimization for various model sizes is summarized here:

TODO

A more thorough write-up of the results can be found [here](./docs/details_Aug2024.md).