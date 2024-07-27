
from datasets import Dataset

import torch
import torch.nn.functional as F


def calc_response_probability(model, tokenizer, example: str,
                              split_str="Assistant:"):
    """
    Estimates the probability of the final response.
    Inputs:
        model:
        tokenizer:
        example: a string of multiple assistant-human interaction steps.
            The last response is to be judged.
        split_str: the string that indicates when the last response begins
    """

    # Split the context and response
    context, response = example.rsplit(split_str, 1)

    # Tokenize everything and move to device
    inputs = tokenizer(example + tokenizer.eos_token, return_tensors='pt')
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # Determine the # of tokens in the response to be judged
    response_tokens = tokenizer(response)
    n_resp_tokens = len(response_tokens.input_ids)

    result = model(**inputs)

    # Calculate the log probability of the response
    # Include the previous token to account for the shift by 1.
    log_probs = F.log_softmax(result.logits[0, -n_resp_tokens-1:, :], dim=-1)
    log_prob = log_probs[
        torch.arange(n_resp_tokens), response_tokens.input_ids
    ].sum()

    return log_prob


def calculate_reference_probs(model, tokenizer, ds, split_str="Assistant:", insert=True):
    # We want to build a cache of \pi_{ref} for every point in the dataset
    # with our fine tuned model

    # What you actually want to do is create a new dataset with a bias field
    # that you can have \log\pi_{ref}(y_W|x) - \log\pi_{ref}(y_L|x) cached
    # So that it can be subtracted in the \log\sigma() term

    def calc_ref_prob_delta(example):
        log_prob_W = calc_response_probability(
            model, tokenizer, example['chosen'], split_str=split_str
        )
        log_prob_L = calc_response_probability(
            model, tokenizer, example['rejected'], split_str=split_str
        )

        if insert:
            example['ref_log_prob_delta'] = log_prob_W - log_prob_L
            example['log_prob_W'] = log_prob_W
            example['log_prob_L'] = log_prob_L
            return example
        else:
            return {
                'ref_log_prob_delta': log_prob_W - log_prob_L,
                'log_prob_W': log_prob_W,
                'log_prob_L': log_prob_L,
            }

    return ds.map(
        calc_ref_prob_delta,
        keep_in_memory=True, load_from_cache_file=False,
        num_proc=1
    )


@torch.no_grad()
def preprocess_example_for_dpo(example: dict, model, tokenizer,
                               split_str="Assistant:"):

    new_example = {}

    # There are some cases where the model generated extra "Assistant:"
    # So we take the earliest last one over the two.
    min_context_len = 1000000000

    for text_key in ('chosen', 'rejected'):
        context, response = example[text_key].rsplit(split_str, 1)
        min_context_len = min(min_context_len, len(context))

    context_idx = min_context_len + len(split_str)

    for text_key in ('chosen', 'rejected'):
        response = example[text_key][context_idx:]

        inputs = tokenizer(
            example[text_key] + tokenizer.eos_token,
            truncation=True, return_tensors='pt'
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        # Determine the # of tokens in the response to be judged
        response_tokens = tokenizer(response)
        n_resp_tokens = len(response_tokens.input_ids)

        result = model(**inputs)

        # Calculate the log probability of the response
        # Include the previous token to account for the shift by 1.
        log_probs = F.log_softmax(result.logits[0, -n_resp_tokens-1:, :], dim=-1)
        log_prob = log_probs[
            torch.arange(n_resp_tokens), response_tokens.input_ids
        ].sum()

        all_tokens = inputs['input_ids'].squeeze().tolist()
        context_idx = len(all_tokens) - n_resp_tokens - 1

        new_example[text_key] = all_tokens
        new_example[text_key + '_start_idx'] = context_idx
        new_example[text_key + '_log_prob'] = log_prob.item()

    return new_example


@torch.no_grad()
def preprocess_dataset_for_dpo(ds: Dataset, model, tokenizer,
                               split_str="Assistant:",
                               save_dir=None) -> Dataset:
    """
    Generate input tokens, split index and probability estimates for a dataset

    Returns:
        train_dataset: which has fields:
            - chosen: All tokens for the human selected example
            - chosen_start_idx: The token start index of the chosen response
            - chosen_log_prob: The reference model probability for the chosen response
            - rejected:
            - rejected_start_idx:
            - rejected_log_prob:
    """

    model.eval()

    def preproc_example(example):
        return preprocess_example_for_dpo(
            example, model, tokenizer, split_str
        )

    ds_new = ds.map(
        preproc_example,
        keep_in_memory=True,
        load_from_cache_file=False,
        num_proc=1
    )

    if save_dir:
        ds_new.save_to_disk(save_dir)

    return ds_new


