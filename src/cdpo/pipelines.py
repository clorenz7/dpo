
import os

from datasets import (
    DatasetDict,
)
import joblib

from cdpo import model_ops
from cdpo import data_utils
from cdpo import evaluation
from cdpo import training
from cdpo import metrics


def dpo_training_pipeline(training_kwargs, model=None, tokenizer=None, device="cuda:0"):

    # Get the model and tokenizer
    if model is None:
        model_kwargs = dict(training_kwargs['model'])
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
    results = training.train_with_dpo(
        model, tokenizer, ds_preproc, training_args
    )

    metrics_file = training_kwargs.get('metrics_file')
    if metrics_file:
        joblib.dump(results, metrics_file)

    save_plot = training_kwargs.get('save_plot')
    if save_plot:
        metrics.plot_validation_curves(
            results, training_args.eval_steps, save_plot=save_plot
        )

    return results


def fine_tuning(training_kwargs, device="cuda:0"):

    # Get the model and tokenizer
    model_kwargs = dict(training_kwargs['model'])
    model_loc = model_kwargs.pop('loc')

    model, tokenizer = model_ops.get_partially_trainable_model(
        model_loc, **model_kwargs
    )
    model.to(device)

    # Get the pre-processed training data
    data_args = dict(training_kwargs['data'])
    ds = data_utils.get_rlhf_data(**data_args)

    # Get trianing args and train
    training_args = training.get_fine_tuning_args(
        **training_kwargs.get('training_args', {})
    )
    results = training.pretrain_on_chosen_response(
        model, tokenizer, ds, training_args
    )

    # Save the metrics file
    metrics_file = training_kwargs.get('metrics_file')
    if metrics_file:
        joblib.dump(results, metrics_file)

    save_plot = training_kwargs.get('save_plot')
    if save_plot:
        metrics.plot_validation_curves(
            results, training_args.eval_steps, save_plot=save_plot
        )

    return results, model, tokenizer
