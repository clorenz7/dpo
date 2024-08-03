
from datasets import load_dataset


def get_rlhf_data(max_chars=1280,  n_train=None, n_valid=None, n_test=None,
                  exclude_test_idxs=None, exclude_train_idxs=None,
                  verbose=0,):
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
        split_dataset = ds['train'].train_test_split(test_size=n_valid)
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
