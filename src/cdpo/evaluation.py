
from datasets import Dataset

import torch
import torch.nn.functional as F

from transformers import DataCollatorWithPadding


from cdpo.data_utils import get_response_start_idx


@torch.no_grad()
def preprocess_example_for_dpo(example: dict, model, collator,
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

    text_keys = ('chosen', 'rejected')

    for text_key in text_keys:
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
def preprocess_batch_example_for_dpo(examples: list, model, collator,
                                     split_str="Assistant:"):
    """
    Uses batch processing to put tokens through model.
    This is slightly faster (approx 20%) on single examples
    But this should open up batch processing
    """
    # There are some cases where the model generated extra "Assistant:"
    # So we take the earliest last one over the two.
    tokenizer = collator.tokenizer

    n_examples = len(examples['chosen'])
    text_keys = ('chosen', 'rejected')
    new_example = {
        'chosen': [None] * n_examples,
        'rejected': [None] * n_examples,
        'chosen_log_prob': [None] * n_examples,
        'rejected_log_prob': [None] * n_examples,
        'response_start_idx': [None] * n_examples,
    }
    all_start_idxs = []
    all_end_idxs = []
    all_inputs = []

    for ex_idx in range(n_examples):
        example = {
            'chosen': examples['chosen'][ex_idx],
            'rejected': examples['rejected'][ex_idx]
        }

        response_start_idx = get_response_start_idx(example, split_str)

        start_idxs = []
        end_idxs = []

        for text_key in text_keys:
            response = example[text_key][response_start_idx:]

            inputs = tokenizer(
                example[text_key] + tokenizer.eos_token,
                truncation=True
            )
            all_inputs.append(inputs)
            new_example[text_key][ex_idx] = inputs.input_ids

            # Determine the # of tokens in the response to be judged
            response_tokens = tokenizer(response + tokenizer.eos_token)
            n_resp_tokens = len(response_tokens.input_ids)
            resp_start_idx = len(inputs.input_ids) - n_resp_tokens
            start_idxs.append(resp_start_idx)
            end_idxs.append(resp_start_idx + n_resp_tokens)
        all_start_idxs.append(start_idxs)
        all_end_idxs.append(end_idxs)

    # Collate inputs, reshape and run through the model
    inputs = collator(all_inputs).to(model.device)
    outputs = model(**inputs)

    for b_idx in range(n_examples):
        # Calculate the log probability of the response
        # Include the previous token to account for the shift by 1.
        start_idx = all_start_idxs[b_idx][0]
        end_idxs = all_end_idxs[b_idx]
        for r_idx, end_idx in enumerate(end_idxs):
            token_log_probs = F.log_softmax(
                outputs.logits[2*b_idx + r_idx, start_idx-1:end_idx-1, :],
                dim=1
            )
            token_labels = inputs['input_ids'][2*b_idx + r_idx, start_idx:end_idx]
            resp_log_prob = token_log_probs[
                torch.arange(token_labels.shape[0]), token_labels
            ].sum()
            new_example[text_keys[r_idx] + '_log_prob'][b_idx] = resp_log_prob.item()

        new_example['response_start_idx'][b_idx] = start_idx

    return new_example


@torch.no_grad()
def preprocess_dataset_for_dpo(ds: Dataset, model, tokenizer,
                               split_str="Assistant:",
                               save_dir=None, batch_size=1) -> Dataset:
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

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer, return_tensors='pt'
    )

    if batch_size == 1:
        def preproc_func(example):
            return preprocess_example_for_dpo(
                example, model, collator, split_str
            )
    else:
        # I have not found this to be faster
        def preproc_func(examples):
            return preprocess_batch_example_for_dpo(
                examples, model, collator, split_str
            )

    ds_new = ds.map(
        preproc_func,
        keep_in_memory=True,
        load_from_cache_file=False,
        num_proc=1,
        batched=batch_size > 1,
        batch_size=batch_size,
        desc="Preprocessing dataset for DPO"
    )

    try:
        if save_dir:
            ds_new.save_to_disk(save_dir)
    except Exception as e:
        print(e)
        import ipdb; ipdb.set_trace()

    return ds_new


