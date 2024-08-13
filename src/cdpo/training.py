
import os
import yaml

import torch
import torch.nn.functional as F

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    TrainerCallback
)

from cdpo.data_utils import get_response_start_idx
from cdpo.data_utils import get_default_dir


class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.metrics = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.metrics.append(logs.copy())
        torch.cuda.empty_cache()


class DataCollatorWithPaddingLabels(DataCollatorWithPadding):
    def __call__(self, features):

        # Pop the labels so they don't error on superclass call
        pad_labels = "labels" in features[0]
        if pad_labels:
            labels = [feature.pop("labels") for feature in features]

        # Pad the input_ids and attention_mask
        batch = super().__call__(features)

        # Pad labels and re-insert
        if pad_labels:
            batch["labels"] = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=-100
            )

        return batch


def get_fine_tuning_args(yaml_file=None, **kwargs):

    train_kwargs = dict(
        num_train_epochs=4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        eval_on_start=True,
        eval_steps=1000,
        logging_steps=1000,
        save_steps=1000,
        save_total_limit=None,
        eval_strategy='steps',
        batch_eval_metrics=True,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        save_only_model=True,
        lr_scheduler_type="linear",
        warmup_ratio=0.002,
        logging_dir=os.path.join(get_default_dir(), "logs"),
    )

    if yaml_file:
        with open(yaml_file, 'r') as fp:
            yaml_params = yaml.safe_load(fp)
        train_kwargs.update(yaml_params)
        if not kwargs.keys().isdisjoint(yaml_params.keys()):
            print(f"Warning! Keyword args {[k for k in kwargs.keys()]} overwrite yaml values!")

    train_kwargs.update(**kwargs)

    training_args = TrainingArguments(**train_kwargs)

    return training_args


def tokenize_and_label_chosen_response(tokenizer, split_str: str, device: str,
                                       example: dict) -> dict:
    """
    Tokenizes the chosen example in the example and labels all tokens except
    the last response as invalid.

    inputs:
        tokenizer:
        split_str: separator for last response (e.g. "Assistant:")
        device: where to move the tensors
        example: data to process
    returns:
        tokenized input
    """

    # Split the context and response
    # context, response = example['chosen'].rsplit(split_str, 1)
    start_idx = get_response_start_idx(example, split_str)
    response = example['chosen'][start_idx:]

    # Tokenize everything and move to device
    inputs = tokenizer(
        example['chosen'] + tokenizer.eos_token,
        truncation=True,
        # max_length=tokenizer.model_max_length, padding="max_length"
    )

    # Determine the # of tokens in the response to be trained on
    response_tokens = tokenizer(response)
    n_resp_tokens = len(response_tokens.input_ids) + 1  # added eos_token

    # Set the label of the context tokens to ignore
    inputs['labels'] = (
        [-100] * (len(inputs['input_ids']) - n_resp_tokens) +
        inputs['input_ids'][-n_resp_tokens:]
    )

    return inputs


def pretrain_on_chosen_response(model, tokenizer, ds,
                                training_args: TrainingArguments,
                                split_str="Assistant:"):
    """
    Pretrain model on the final response of the chosen text
    Inputs:
        model:
        tokenizer:
        ds: dataset to train on
        training_args:
        split_str:
    Returns:
        trainer: Trainer class
        tokenized_dataset:
    """

    # Tokenize the dataset
    def tokenize_func(example):
        return tokenize_and_label_chosen_response(
            tokenizer, split_str, model.device, example
        )

    tokenized_dataset = ds.map(tokenize_func, batched=False)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'labels'])

    # TODO: Try one of the build-in DataCollators
    data_collator = DataCollatorWithPaddingLabels(
        tokenizer=tokenizer,
        return_tensors='pt'
    )

    metrics_callback = MetricsCallback()

    model.train()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['valid'],
        data_collator=data_collator,
        callbacks=[metrics_callback],
    )

    trainer.train()

    return metrics_callback.metrics


class DpoTrainer(Trainer):
    """
    Trainer subclass which handles DPO loss calculation
    """

    def __init__(self, *args, beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False):

        start_idx = inputs.pop('start_idx')
        end_idx = inputs.pop('end_idx')
        ref_log_probs = inputs.pop('log_probs')

        outputs = model(**inputs)

        log_probs = []

        for b_idx in range(start_idx.shape[0]):
            batch_probs = []
            for ex_idx in range(2):  # Chosen, then rejected example
                st_idx = start_idx[b_idx, ex_idx]
                e_idx = end_idx[b_idx, ex_idx]
                token_log_probs = F.log_softmax(
                    outputs.logits[b_idx, ex_idx, st_idx-1:e_idx-1, :],
                    dim=1
                )
                token_labels = inputs['input_ids'][b_idx, ex_idx, st_idx:e_idx]
                response_log_prob = token_log_probs[
                    torch.arange(token_labels.shape[0]), token_labels
                ].sum()
                batch_probs.append(response_log_prob)
            log_probs.append(torch.stack(batch_probs))

        ref_delta = ref_log_probs[:, 0] - ref_log_probs[:, 1]
        log_probs_T = torch.stack(log_probs)
        resp_delta = log_probs_T[:, 0] - log_probs_T[:, 1]

        loss = -F.logsigmoid(self.beta * (resp_delta - ref_delta)).mean()

        if return_outputs:
            # Cache the computed log probabilities for metrics calculations
            outputs['log_probs'] = log_probs_T
            outputs['ref_log_probs'] = ref_log_probs
            return loss, outputs
        else:
            return loss


class EvalMetricsAggregator:
    """
    Stores batch results and computes final evaluation metrics for DPO
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.win_flags = []
        self.log_prob_deltas = []
        self.rewards_W = []
        self.rewards_L = []

    def compute_metrics(self, outputs: EvalPrediction, compute_result=False):

        # Compute win flags and log_prob_deltas
        log_probs = outputs.predictions[1]
        ref_log_probs = outputs.predictions[2]
        self.win_flags.extend(
            (log_probs[:, 0] > log_probs[:, 1]).to(torch.float32).cpu().tolist()
        )
        self.log_prob_deltas.extend(
            (log_probs[:, 0] - log_probs[:, 1]).cpu().tolist()
        )
        self.rewards_W.extend(
            (log_probs[:, 0] - ref_log_probs[:, 0]).cpu().tolist()
        )
        self.rewards_L.extend(
            (log_probs[:, 1] - ref_log_probs[:, 1]).cpu().tolist()
        )

        if compute_result:
            n_batch = len(self.win_flags)
            metrics = {
                'win_rate': 100.0 * sum(self.win_flags) / n_batch,
                'avg_log_prob_delta': sum(self.log_prob_deltas) / n_batch,
                'avg_reward_delta': (sum(self.rewards_W) - sum(self.rewards_L)) / n_batch,
            }
            self.clear()
            return metrics

        return {}


_global_metrics_state = EvalMetricsAggregator()


def compute_metrics(outputs: EvalPrediction, compute_result=False):
    return _global_metrics_state.compute_metrics(outputs, compute_result)


def reset_metrics():
    _global_metrics_state.clear()


class DataCollatorDpo(DataCollatorWithPadding):

    def __call__(self, features):

        input_ids = []
        start_idx, end_idx = [], []
        log_probs = []
        for example in features:
            # We flatten input_ids so that the parent class padding works below
            input_ids.append(example['chosen'])
            input_ids.append(example['rejected'])
            start_idx.append([example['response_start_idx']]*2)

            end_idx.append(
                [len(example['chosen']), len(example['rejected'])]
            )

            log_probs.append(
                [example['chosen_log_prob'], example['rejected_log_prob']]
            )

        # Pad the input_ids and attention_mask
        batch = super().__call__({'input_ids': input_ids})
        # Reshape the flattened tensor to put chosen / rejected in its own dimension
        new_shape = (len(end_idx), 2, batch['input_ids'].shape[-1])
        batch['input_ids'] = batch['input_ids'].view(new_shape)
        batch['attention_mask'] = batch['attention_mask'].view(new_shape)

        batch['start_idx'] = torch.tensor(start_idx)
        batch['end_idx'] = torch.tensor(end_idx)
        batch['log_probs'] = torch.tensor(log_probs)

        return batch


def get_dpo_training_args(yaml_file=None, **kwargs):

    train_kwargs = dict(
        label_names=['start_idx', 'end_idx', 'log_probs'],
        eval_on_start=True,
        eval_steps=100,
        logging_steps=100,
        save_steps=100,
        save_total_limit=None,
        eval_strategy='steps',
        batch_eval_metrics=True,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        save_only_model=True,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=125,
        optim="rmsprop",  # What they used in paper
        logging_dir=os.path.join(get_default_dir(), "logs"),
        per_device_train_batch_size=1,
    )

    if yaml_file:
        with open(yaml_file, 'r') as fp:
            yaml_params = yaml.safe_load(fp)
        train_kwargs.update(yaml_params)
        if not kwargs.keys().isdisjoint(yaml_params.keys()):
            print(f"Warning! Keyword args {[k for k in kwargs.keys()]} overwrite yaml values!")

    train_kwargs.update(**kwargs)

    training_args = TrainingArguments(**train_kwargs)

    return training_args



def train_with_dpo(model, tokenizer, ds_preproc, training_args):

    data_collator = DataCollatorDpo(
        return_tensors='pt',
        tokenizer=tokenizer,
    )

    # Setup the aggregator of metrics across all batches in the eval set
    agg = EvalMetricsAggregator()
    # Instantiate the callback to record progress
    metrics_callback = MetricsCallback()

    trainer = DpoTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ds_preproc['train'],
        eval_dataset=ds_preproc['valid'],
        compute_metrics=agg.compute_metrics,
        callbacks=[metrics_callback]
    )
    trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )

    return metrics_callback.metrics
