from functools import partial

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

    return ds.map(calc_ref_prob_delta, keep_in_memory=True, load_from_cache_file=False, num_proc=1)
