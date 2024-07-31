
from datasets import load_dataset


def get_rlhf_data(max_chars=1280, verbose=0, n_train=None, n_test=None,
                  exclude_test_idxs=None, exclude_train_idxs=None):
    ds = load_dataset("Anthropic/hh-rlhf")

    if max_chars is not None:
        def filter_func(x):
            return (
                len(x['chosen']) <= max_chars and
                len(x['rejected']) <= max_chars
            )

        ds['train'] = ds['train'].filter(filter_func)
        ds['test'] = ds['test'].filter(filter_func)

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
        print(f"Test data has {len(ds['test'])} points")

    return ds
