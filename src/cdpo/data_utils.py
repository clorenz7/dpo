import os

from datasets import load_dataset
import torch

DEFAULT_DIR = os.path.join(os.path.expanduser('~'), 'cdpo_results')


def get_default_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def get_default_dir():
    default_dir = os.environ.get('CDPO_DEFAULT_DIR', DEFAULT_DIR)
    os.makedirs(default_dir, exist_ok=True)

    return default_dir


def get_response_start_idx(example, split_str):

    # There are some cases where the model generated extra "Assistant:"
    # So we take the earliest last one over the two.
    min_context_len = 1000000000

    for text_key in ('chosen', 'rejected'):
        context, response = example[text_key].rsplit(split_str, 1)
        min_context_len = min(min_context_len, len(context))

    response_start_idx = min_context_len + len(split_str)

    return response_start_idx


def get_rlhf_data(max_chars=1280,  n_train=None, n_valid=None, n_test=None,
                  exclude_test_idxs=None, exclude_train_idxs=None,
                  verbose=0, seed=None, validate_on_test=False):
    """
    Inputs:
        validate_on_test: Pull the validation data from the test set.
            This is useful when finetuning before training DPO so that
            you can split the validation set from the training set when
            doing DPO. Otherwise, there is a distribution shift which
            throws off the validation metrics.
    """
    ds = load_dataset("Anthropic/hh-rlhf")

    if max_chars is not None:
        def filter_func(x):
            return (
                len(x['chosen']) <= max_chars and
                len(x['rejected']) <= max_chars
            )

        ds['train'] = ds['train'].filter(filter_func)
        ds['test'] = ds['test'].filter(filter_func)

    if n_valid:
        if validate_on_test:
            ds['valid'] = ds['test'].select(range(n_valid))
        else:
            split_dataset = ds['train'].train_test_split(test_size=n_valid, seed=seed)
            ds['train'] = split_dataset['train']
            ds['valid'] = split_dataset['test']

    if n_train:
        ds['train'] = ds['train'].select(range(n_train))
    if n_test:
        ds['test'] = ds['test'].select(range(n_test))

    if exclude_test_idxs:
        idxs = [
            i for i in range(len(ds['test'])) if i not in exclude_test_idxs
        ]
        ds['test'] = ds['test'].select(idxs)

    if exclude_train_idxs:
        idxs = [
            i for i in range(len(ds['train'])) if i not in exclude_train_idxs
        ]
        ds['train'] = ds['train'].select(idxs)

    if verbose:
        print(f"Training data has {len(ds['train'])} points")
        if n_valid:
            print(f"Validation data has {len(ds['valid'])} points")
        print(f"Test data has {len(ds['test'])} points")

    return ds
