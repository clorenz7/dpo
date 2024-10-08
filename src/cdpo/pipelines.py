
import os
import yaml

from datasets import (
    DatasetDict,
)
import joblib

from cdpo import model_ops
from cdpo import data_utils
from cdpo import evaluation
from cdpo import training
from cdpo import metrics


def dpo_training_pipeline(training_kwargs, save_dir, model=None, tokenizer=None, device="cuda:0"):

    # Get the model and tokenizer
    if model is None:
        print("Loading Model for DPO...")
        model_kwargs = dict(training_kwargs['model'])
        model_dir = model_kwargs.pop('loc')

        model, tokenizer = model_ops.get_partially_trainable_model(
            model_dir, **model_kwargs
        )
        model.to(device)

    # Get the pre-processed training data
    data_args = dict(training_kwargs['data'])

    pp_data_dir = data_args.pop('loc', '')
    max_tokens = data_args.pop('max_tokens', -1)
    if not pp_data_dir:
        pp_data_dir = os.path.join(save_dir, "dpo_preproc_data")

    if os.path.exists(pp_data_dir):
        print(f"Loading preprocessing data from {pp_data_dir}")
        ds_preproc = DatasetDict.load_from_disk(pp_data_dir)
    else:
        print("Pre-processing data for DPO...")
        ds = data_utils.get_rlhf_data(**data_args)
        ds_preproc = evaluation.preprocess_dataset_for_dpo(
            ds, model, tokenizer,
            save_dir=pp_data_dir
        )

    if max_tokens > 0:
        ds_preproc['train'] = data_utils.limit_num_tokens(
            ds_preproc['train'], max_tokens
        )

    # Get the training arguments and train the model
    output_dir = os.path.join(save_dir, "dpo")
    training_args = training.get_dpo_training_args(
        output_dir=output_dir,
        **training_kwargs.get('training_args', {})
    )
    print(f"Training with DPO ({output_dir})...")
    results = training.train_with_dpo(
        model, tokenizer, ds_preproc, training_args
    )

    metrics_file = os.path.join(save_dir, "dpo_training_metrics.joblib")
    joblib.dump(results, metrics_file)

    plot_path = os.path.join(save_dir, "dpo_validation_curves.png")
    metrics.plot_validation_curves(
        results, training_args.eval_steps, save_plot=plot_path,
        title=os.path.basename(save_dir),
    )
    print(f"Validation curve saved to {plot_path}")

    return results


def fine_tuning(training_kwargs, save_dir, device="cuda:0"):

    # Get the model and tokenizer
    print("Loading Model...")
    model_kwargs = dict(training_kwargs['model'])
    model_loc = model_kwargs.pop('loc')

    model, tokenizer = model_ops.get_partially_trainable_model(
        model_loc, **model_kwargs
    )
    model.to(device)

    # Get the pre-processed training data
    print("Loading Finetuning Data...")
    data_args = dict(training_kwargs['data'])
    ds = data_utils.get_rlhf_data(**data_args)

    # Get training args and train
    output_dir = os.path.join(save_dir, "sft")
    training_args = training.get_fine_tuning_args(
        output_dir=output_dir,
        **training_kwargs.get('training_args', {})
    )

    print(f"Finetuning ({output_dir}) ...")
    results = training.pretrain_on_chosen_response(
        model, tokenizer, ds, training_args
    )

    # Save the metrics file
    metrics_file = os.path.join(save_dir, "sft_training_metrics.joblib")
    joblib.dump(results, metrics_file)

    plot_path = os.path.join(save_dir, "sft_validation_curves.png")
    metrics.plot_validation_curves(
        results, training_args.eval_steps, save_plot=plot_path,
        title=os.path.basename(save_dir),
    )
    print(f"Validation curve saved to {plot_path}")

    return results, model, tokenizer


def sft_and_dpo(params_file, results_dir, device=None):

    # Get parameters, device, and setup output directory
    if not device:
        device = data_utils.get_default_device()

    with open(params_file, 'r') as fp:
        train_params = yaml.safe_load(fp)

    exp_name = os.path.splitext(os.path.basename(params_file))[0]
    output_dir = os.path.join(results_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # Finetune basemodel (if necessary)
    ft_params = train_params.get('fine_tuning')
    if ft_params:
        # TODO: Evaluate the basemodel chosen vs reject before and after
        # TODO: Implement early stopping?
        sft_result, model, tokenizer = fine_tuning(
            ft_params, output_dir, device=device
        )
    else:
        sft_result = model = tokenizer = None

    # Do DPO
    dpo_result = dpo_training_pipeline(
        train_params['dpo'], output_dir,
        model=model, tokenizer=tokenizer,
        device=device
    )

    return sft_result, dpo_result
