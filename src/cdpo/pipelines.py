
import os

from datasets import (
    DatasetDict,
)

from cdpo import model_ops
from cdpo import data_utils
from cdpo import evaluation
from cdpo import training


def dpo_training_pipeline(training_kwargs, model=None, tokenizer=None, device="cuda:0"):

    # Get the model and tokenizer
    if model is None:
        model_kwargs = dict(training_kwargs.get('model', {}))
        model_dir = model_kwargs.pop('loc')

        model, tokenizer = model_ops.get_partially_trainable_model(
            model_dir, **model_kwargs
        )
        model.to(device)

    # Get the pre-processed training data
    data_args = dict(training_kwargs['data'])
    save_dir = data_args.pop('save_dir', '')
    if os.path.exists(save_dir):
        ds_preproc = DatasetDict.load_from_disk(save_dir)
    else:
        ds = data_utils.get_rlhf_data(**data_args)
        ds_preproc = evaluation.preprocess_dataset_for_dpo(
            ds, model, tokenizer, save_dir=save_dir
        )

    # Get the training arguments and train the model
    training_args = training.get_dpo_training_args(
        **training_kwargs.get('training_args', {})
    )
    trainer = training.train_with_dpo(
        model, tokenizer, ds_preproc, training_args
    )

    return trainer
