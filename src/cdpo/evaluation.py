
from datasets import Dataset

import torch
import torch.nn.functional as F


from cdpo.data_utils import get_response_start_idx


def calc_response_probability(model, tokenizer, example: str,
                              start_idx=None, split_str="Assistant:"):
    """
    Estimates the probability of the final response.
    Inputs:
        model:
        tokenizer:
        example: a string of multiple assistant-human interaction steps.
            The last response is to be judged.
        start_idx: The string index where the response starts
        split_str: The string that indicates when the last response begins.
            Only used if start_idx is None
    """

    # Split the context and response
    if start_idx is None:
        context, response = example.rsplit(split_str, 1)
    else:
        response = example[start_idx:]

    # Tokenize everything and move to device
    inputs = tokenizer(example + tokenizer.eos_token, return_tensors='pt')
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # Determine the # of tokens in the response to be judged
    response_tokens = tokenizer(response + tokenizer.eos_token)
    n_resp_tokens = len(response_tokens.input_ids)

    result = model(**inputs)

    # Calculate the log probability of the response
    # Include the previous token to account for the shift by 1.
    log_probs = F.log_softmax(result.logits[0, -n_resp_tokens-1:, :], dim=-1)
    log_prob = log_probs[
        torch.arange(n_resp_tokens), response_tokens.input_ids
    ].sum()

    return log_prob, inputs, n_resp_tokens


def calculate_reference_probs(model, tokenizer, ds, split_str="Assistant:", insert=True):
    # We want to build a cache of \pi_{ref} for every point in the dataset
    # with our fine tuned model

    # What you actually want to do is create a new dataset with a bias field
    # that you can have \log\pi_{ref}(y_W|x) - \log\pi_{ref}(y_L|x) cached
    # So that it can be subtracted in the \log\sigma() term

    def calc_ref_prob_delta(example):

        response_start_idx = get_response_start_idx(example, split_str)

        log_prob_W, _, _ = calc_response_probability(
            model, tokenizer, example['chosen'],
            start_idx=response_start_idx
        )
        log_prob_L, _, _ = calc_response_probability(
            model, tokenizer, example['rejected'],
            start_idx=response_start_idx
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
    start_idxs = []
    response_start_idx = get_response_start_idx(example, split_str)

    # TODO: Put these in a batch and run them through the model
    for text_key in ('chosen', 'rejected'):
        response = example[text_key][response_start_idx:]

        inputs = tokenizer(
            example[text_key] + tokenizer.eos_token,
            truncation=True, return_tensors='pt'
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        # Determine the # of tokens in the response to be judged
        response_tokens = tokenizer(response + tokenizer.eos_token)
        n_resp_tokens = len(response_tokens.input_ids)

        result = model(**inputs)

        # Calculate the log probability of the response
        # Include the previous token to account for the shift by 1.
        log_probs = F.log_softmax(result.logits[0, -n_resp_tokens-1:, :], dim=-1)
        log_prob = log_probs[
            torch.arange(n_resp_tokens), response_tokens.input_ids
        ].sum()

        all_tokens = inputs['input_ids'].squeeze().tolist()
        resp_start_idx = len(all_tokens) - n_resp_tokens
        start_idxs.append(resp_start_idx)
        new_example[text_key] = all_tokens
        new_example[text_key + '_log_prob'] = log_prob.item()

    new_example['response_start_idx'] = start_idxs[0]

    if start_idxs[0] != start_idxs[1]:
        print("Token start:", start_idxs)
        print("Char Start:", response_start_idx)
        print("CHOSEN:", example['chosen'])
        print("REJECTED:", example['rejected'])

    return new_example


@torch.no_grad()
def preprocess_example_for_dpo2(example: dict, model, collator,
                                split_str="Assistant:"):
    """
    Uses batch processing to put tokens through model.
    This is slightly faster (approx 20%) on single examples
    But this should open up batch processing
    """
    # There are some cases where the model generated extra "Assistant:"
    # So we take the earliest last one over the two.
    tokenizer = collator.tokenizer
    response_start_idx = get_response_start_idx(example, split_str)

    new_example = {}
    start_idxs = []
    end_idxs = []
    all_inputs = []

    for text_key in ('chosen', 'rejected'):
        response = example[text_key][response_start_idx:]

        inputs = tokenizer(
            example[text_key] + tokenizer.eos_token,
            truncation=True
        )
        all_inputs.append(inputs)
        new_example[text_key] = inputs.input_ids

        # Determine the # of tokens in the response to be judged
        response_tokens = tokenizer(response + tokenizer.eos_token)
        n_resp_tokens = len(response_tokens.input_ids)
        resp_start_idx = len(inputs.input_ids) - n_resp_tokens
        start_idxs.append(resp_start_idx)
        end_idxs.append(resp_start_idx + n_resp_tokens)

    inputs = collator(all_inputs).to(model.device)
    outputs = model(**inputs)

    # Calculate the log probability of the response
    # Include the previous token to account for the shift by 1.
    start_idx = start_idxs[0]
    for b_idx, end_idx in enumerate(end_idxs):
        token_log_probs = F.log_softmax(
            outputs.logits[b_idx, start_idx-1:end_idx-1, :],
            dim=1
        )
        token_labels = inputs['input_ids'][b_idx, start_idx:end_idx]
        resp_log_prob = token_log_probs[
            torch.arange(token_labels.shape[0]), token_labels
        ].sum()
        new_example[text_keys[b_idx] + '_log_prob'] = resp_log_prob.item()

    new_example['response_start_idx'] = start_idx

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

    try:
        if save_dir:
            ds_new.save_to_disk(save_dir, )
    except Exception as e:
        print(e)

    return ds_new


