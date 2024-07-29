
import torch
import torch.nn.functional as F

from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


BASE_DIR = r"D:\training\cdpo"


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
    context, response = example['chosen'].rsplit(split_str, 1)

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

    model.train()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    return trainer, tokenized_dataset


class DpoTrainer(Trainer):

    def compute_loss_split(self, model, inputs, return_outputs=False):

        ref_delta = inputs['rejected_log_prob'] - inputs['chosen_log_prob']

        # Run the model on the context
        start_idx = inputs['chosen_start_idx'].item()
        ctx_input = {'input_ids': inputs['chosen'][:, :start_idx]}
        outputs_ctx = model(**ctx_input)

        # Grab the transformer state
        past_key_values = outputs_ctx['past_key_values']

        # Run model on chosen response with state
        # TODO: make this a for loop over the two values
        labels_W = inputs['chosen'][:, start_idx:]
        input_W = {
            'input_ids': labels_W,
            'past_key_values': past_key_values
        }
        outputs_W = model(**input_W)

        # Run model on rejected response with state
        labels_L = inputs['rejected'][:, start_idx:]
        input_L = {
            'input_ids': labels_L,
            'past_key_values': past_key_values
        }
        outputs_L = model(**input_L)

        # Estimate probability of chosen and rejected
        # BE CAREFUL WITH OFF BY ONE!
        logits_W = torch.cat(
            (outputs_ctx.logits[:, -1:, :], outputs_W.logits[:, :-1, :]),
            dim=1
        )
        n_W = labels_W.shape[1]
        logprobs_W = F.log_softmax(logits_W, dim=2)
        log_prob_W = logprobs_W[0, torch.arange(n_W), labels_W[0, :]].sum()

        logits_L = torch.cat(
            (outputs_ctx.logits[:, -1:, :], outputs_L.logits[:, :-1, :]),
            dim=1
        )
        n_L = labels_L.shape[1]
        logprobs_L = F.log_softmax(logits_L, dim=2)
        log_prob_L = logprobs_L[0, torch.arange(n_L), labels_L[0, :]].sum()

        # Calculate the loss with logsigmoid and the reference probs
        beta = 0.1
        loss = -F.logsigmoid(beta * (log_prob_W - log_prob_L + ref_delta)).mean()

        if return_outputs:
            return loss, (outputs_ctx, outputs_W, outputs_L)
        else:
            return loss

    def compute_loss(self, model, inputs, return_outputs=False):

        beta = 0.1

        start_idx = inputs.pop('start_idx')
        end_idx = inputs.pop('end_idx')
        ref_log_probs = inputs.pop('log_probs')

        outputs = model(**inputs)

        log_probs = []

        all_token_log_probs = F.log_softmax(
            outputs.logits, dim=2
        )

        for b_idx in range(start_idx.shape[0]):
            st_idx = start_idx[b_idx]
            e_idx = end_idx[b_idx]
            # token_log_probs = F.log_softmax(
            #     outputs.logits[b_idx, st_idx:e_idx-1, :],
            #     dim=1
            # )
            token_labels = inputs['input_ids'][b_idx, st_idx+1:e_idx]
            # resp_log_prob = token_log_probs[torch.arange(token_labels.shape[0]), token_labels].sum()
            resp_log_prob = all_token_log_probs[b_idx, torch.arange(st_idx, e_idx-1), token_labels].sum()
            log_probs.append(resp_log_prob)

        ref_delta = ref_log_probs[::2] - ref_log_probs[1::2]
        log_probs_T = torch.stack(log_probs)
        resp_delta = log_probs_T[::2] - log_probs_T[1::2]

        loss = -F.logsigmoid(beta * (resp_delta - ref_delta)).mean()

        if return_outputs:
            return loss, outputs
        else:
            return loss


class DataCollatorDpo(DataCollatorWithPadding):

    def __call__(self, features):

        input_ids = []
        start_idx, end_idx = [], []
        log_probs = []
        for example in features:
            input_ids.append(example['chosen'])
            input_ids.append(example['rejected'])

            start_idx.append(example['response_start_idx'])
            start_idx.append(example['response_start_idx'])
            # TODO: This is only for testing
            # start_idx.append(example['chosen_start_idx'])
            # start_idx.append(example['rejected_start_idx'])

            end_idx.append(len(example['chosen']))
            end_idx.append(len(example['rejected']))

            log_probs.append(example['chosen_log_prob'])
            log_probs.append(example['rejected_log_prob'])

        # Pad the input_ids and attention_mask
        batch = super().__call__({'input_ids': input_ids})

        batch['start_idx'] = torch.tensor(start_idx)
        batch['end_idx'] = torch.tensor(end_idx)
        batch['log_probs'] = torch.tensor(log_probs)

        return batch


def train_with_dpo(model, ds, training_args):

    data_collator = DataCollatorDpo(
        return_tensors='pt'
    )

    model.train()
    trainer = DpoTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    trainer.train()
