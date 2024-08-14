import os
from unittest import TestCase

from datasets import (
    Dataset,
    DatasetDict,
    load_dataset
)
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments
)

from base import TestModelBase
import cdpo
from cdpo import training
from cdpo import model_ops
from cdpo import evaluation
from cdpo.data_utils import get_default_dir


EXAMPLE = """Human: What are some cuss words in english?

Assistant: Here’s an incomplete list.

Ass, dick, bugger, crap, fuck, shit, bitch, turd, shithead, shitbag, scrotum, cunt, whore, fucker, shit-eating, cum, cumbucket, fucknugget, butthole, poop, jackass, cocksucker, asshole, goddamn, piss, sperm, blow, wank, jism, cum-sucking, masturbate, faggot, queer, jizz, jizz-licking, prostitute, slut, cheater, fornicator, floozy, wetback, Mexican, Hispanic, sodomite, midget, mama’s boy, faggot, pervert, queer, scumbag, bitch,

Human: What's your favorite one?

Assistant: I haven't even thought about it."""


def test_tokenize_and_label():
    split_str = "Assistant:"
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    inputs = training.tokenize_and_label_chosen_response(
        tokenizer, split_str, device, {'chosen': EXAMPLE}
    )

    # Make sure that we added a labels to the input
    assert 'input_ids' in inputs
    assert 'labels' in inputs

    assert len(inputs['input_ids']) == 201
    assert len(inputs['labels']) == 201

    # The first few tokens should be invalid
    assert inputs['labels'][:5] == [-100]*5

    # The last 9 tokens should be the same
    assert inputs['labels'][-9:] == inputs['input_ids'][-9:]


class TestTraining(TestModelBase):

    def test_pre_train(self):

        # Just run a short training to make sure it works
        training_args = TrainingArguments(
            output_dir=os.path.join(self.BASE_DIR, "results_smoke"),
            overwrite_output_dir=True,
            max_steps=16,
            per_device_train_batch_size=4,
            save_steps=16,
            save_total_limit=None,
            logging_dir=os.path.join(self.BASE_DIR, "logs"),
            learning_rate=1e-9,
            lr_scheduler_type="constant",
        )

        self.model.train()
        training.pretrain_on_chosen_response(
            self.model, self.tokenizer, self.ds['train'], training_args
        )


class TestDpoTraining(TestCase):

    BASE_DIR = get_default_dir()

    def test_dpo_train(self):
        # data_dir = r'D:\training\cdpo\datasets\dpo_preproc_gpt2sm_jul26_smoke'
        data_dir = r'D:\training\cdpo\datasets\dpo_preproc_gpt2sm_jul29'
        ds_train = Dataset.load_from_disk(data_dir)

        model_dir = r'D:\training\cdpo\results_jul20\checkpoint-45582'

        model, tokenizer = model_ops.get_partially_trainable_model(
            model_dir, n_layers_freeze=0, dropout=0.1,
            tokenizer_name="openai-community/gpt2",
        )
        model.to('cuda:0')

        training_args = TrainingArguments(
            output_dir=os.path.join(self.BASE_DIR, "results_dpo"),
            overwrite_output_dir=True,
            max_steps=16*10,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            save_steps=16*10,
            save_total_limit=None,
            logging_dir=os.path.join(self.BASE_DIR, "logs"),
            learning_rate=5e-5,
            lr_scheduler_type="linear",
            remove_unused_columns=False
        )

        model.train()

        data_collator = training.DataCollatorDpo(
            return_tensors='pt',
            tokenizer=tokenizer,
        )

        trainer = training.DpoTrainer(
            model=model,
            args=training_args,
            train_dataset=ds_train,
            data_collator=data_collator,
        )

        trainer.train()


    def test_dpo_prob_calcs(self):
        model_dir = r'D:\training\cdpo\results_jul20\checkpoint-45582'
        model, tokenizer = model_ops.get_partially_trainable_model(
            model_dir, n_layers_freeze=0, dropout=0.0,  # dropout=0.1,
            tokenizer_name="openai-community/gpt2",
        )
        model.to('cuda:0')

        # Load and preprocess the dataset but only one element
        ds = load_dataset("Anthropic/hh-rlhf")
        ds['train'] = ds['train'].select(range(1))

        ds_train = evaluation.preprocess_dataset_for_dpo(
            ds['train'], model, tokenizer
        )

        # Setup a Trainer
        model.train()
        training_args = TrainingArguments(
            output_dir=os.path.join(self.BASE_DIR, "results_dpo"),
            overwrite_output_dir=True,
            max_steps=4 * 4,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            save_steps=16*10,
            save_total_limit=None,
            logging_dir=os.path.join(self.BASE_DIR, "logs"),
            learning_rate=1e-9,
            lr_scheduler_type="linear",
            remove_unused_columns=False
        )
        data_collator = training.DataCollatorDpo(
            return_tensors='pt',
            tokenizer=tokenizer,
        )
        trainer = training.DpoTrainer(
            model=model,
            args=training_args,
            train_dataset=ds_train,
            data_collator=data_collator,
        )

        # Compute loss on first batch of data
        first_batch = next(iter(trainer.get_train_dataloader()))
        loss, outputs = trainer.compute_loss(model, first_batch, True)

        # Ensure that the precalculated log_probs and the model log_probs
        # are roughly the same
        assert torch.allclose(outputs.ref_log_probs, outputs.log_probs)


    def test_validation_set(self):
        save_dir = r'D:\training\cdpo\datasets\dpo_preproc_gpt2sm_aug02'
        ds_preproc = DatasetDict.load_from_disk(save_dir)

        model_dir = r'D:\training\cdpo\results_jul20\checkpoint-45582'

        model, tokenizer = model_ops.get_partially_trainable_model(
            model_dir, n_layers_freeze=0, dropout=0.1,
            tokenizer_name="openai-community/gpt2",
        )
        model.to('cuda:0')

        training_args = TrainingArguments(
            output_dir=os.path.join(self.BASE_DIR, "results_smoke"),
            overwrite_output_dir=True,
            max_steps=800,
            # num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=64,
            save_steps=100,
            save_total_limit=None,
            logging_dir=os.path.join(self.BASE_DIR, "logs"),
            learning_rate=3e-6,
            logging_steps=100,
            warmup_steps=125,
            # adam_beta1=0.0,  # Make it roughly RMSProp
            # adam_beta2=0.9,
            # warmup_ratio=0.002,
            # lr_scheduler_type="linear",
            remove_unused_columns=False,
            eval_do_concat_batches=False,
            lr_scheduler_type="constant_with_warmup",
            # lr_scheduler_type="constant",
            batch_eval_metrics=True,
            label_names=['start_idx', 'end_idx', 'log_probs'],
            eval_on_start=True,
            eval_steps=100,
            eval_strategy='steps',
        )

        data_collator = cdpo.training.DataCollatorDpo(
            return_tensors='pt',
            tokenizer=tokenizer,
        )

        metrics_state = cdpo.training.EvalMetricsAggregator()

        def compute_metrics(outputs, compute_result):
            return metrics_state.compute_metrics(outputs, compute_result)

        model.train()
        trainer = cdpo.training.DpoTrainer(
            model=model,
            args=training_args,
            train_dataset=ds_preproc['train'],
            eval_dataset=ds_preproc['valid'],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        # result = trainer.evaluate(ds_preproc['valid'].select(range(100)))
        # result_pred = trainer.predict(ds_preproc['valid'].select(range(12)))

        import ipdb; ipdb.set_trace()
