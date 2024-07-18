
import os

from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


BASE_DIR = r"D:\training\cdpo"


def tokenize_and_label_chosen_response(tokenizer, split_str, device, example):
    # Split the context and response
    context, response = example['chosen'].rsplit(split_str, 1)

    # Tokenize everything and move to device
    inputs = tokenizer(example['chosen'] + tokenizer.eos_token, return_tensors='pt')
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Determine the # of tokens in the response to be judged
    response_tokens = tokenizer(response)
    n_resp_tokens = len(response_tokens.input_ids)

    inputs['labels'] = inputs['input_ids'].clone()
    inputs['labels'][0, :-(n_resp_tokens+1)] = -100

    return inputs


def pretrain_on_chosen(model, tokenizer, ds, n_steps, split_str="Assistant:"):

    def tokenize_func(example):
        # context, response = example['chosen'].rsplit(split_str, 1)
        # context = tokenizer(context)
        # inputs = tokenizer(example + tokenizer.eos_token, return_tensors='pt')
        return tokenizer(example['chosen'], truncation=True, max_length=2048)

    tokenized_dataset = ds.map(tokenize_func, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids'])

    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, "results"),
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=200,
        save_total_limit=4,
        logging_dir=os.path.join(BASE_DIR, "logs"),
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()


def train_with_dpo(model, ds, n_steps, split_str="Assistant:"):
    pass