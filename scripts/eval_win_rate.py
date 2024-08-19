import argparse

from datasets import (
    load_dataset,
    Dataset
)
import numpy as np
from transformers import (
    AutoModelForCausalLM
)

from cdpo import preference_eval
from cdpo import model_ops


def main(args):
    if args.gen_dir:
        ds2 = Dataset.load_from_disk(args.gen_dir)
    else:
        # Get the model
        print("Loading the model...")
        tokenizer = model_ops.get_tokenizer("openai-community/gpt2")
        model = AutoModelForCausalLM.from_pretrained(args.model)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        model.to(args.device)

        # Load the dataset
        print("Loading the dataset...")
        n_samples = args.n_test
        if args.exclude:
            exclude_idxs = set(int(x) for x in args.exclude.split(","))
            keep_idxs = set(range(n_samples + len(exclude_idxs)))
            keep_idxs = list(keep_idxs.difference(exclude_idxs))
            if len(keep_idxs) > n_samples:
                keep_idxs = sorted(keep_idxs)[:n_samples]
        else:
            keep_idxs = range(n_samples)

        max_chars = 1280
        ds = load_dataset("Anthropic/hh-rlhf")
        # ds['train'] = ds['train'].filter(lambda x: len(x['chosen']) <= max_chars)
        ds['test'] = ds['test'].filter(lambda x: len(x['chosen']) <= max_chars)
        ds['test'] = ds['test'].select(keep_idxs)

        print("Generating responses...")
        ds2 = preference_eval.generate_responses(
            model, tokenizer,
            ds['test'], device=args.device,
            n_responses=args.n_trials,
        )

        if args.output_dir:
            ds2.save_to_disk(args.output_dir)

    print("Evaluating responses...")
    win_rates = []
    for trial_idx in range(args.n_trials):
        win_rate, fail_idxs, win_flags = preference_eval.evaluate_win_rate(
            ds2, key_1='new_responses', trial_idx=trial_idx
        )
        print(f"---- Trial #{trial_idx+1} ----")
        print(f"Win Rate: {win_rate*100:0.1f}%")
        if fail_idxs:
            print(f"Fail indexes: {fail_idxs}")
        win_rates.append(win_rate)

    if args.n_trials > 0:
        avg_win_rate = np.mean(win_rates)
        std_win_rate = np.std(win_rates)
        print(f"Avg Win Rate: {avg_win_rate*100:0.1f}% STD: {std_win_rate*100:0.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate the win rate for a given model'
    )
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='Model name or directory path to load'
    )
    parser.add_argument(
        '-o', '--output_dir', type=str,
        help='Path to the output directory to save generated dataset'
    )
    parser.add_argument(
        '-g', '--gen_dir', type=str,
        help='Path to the generated dataset directory'
    )
    parser.add_argument(
        '-n', '--n_test', type=int, default=100,
        help='Number of test samples to generate and evaluate'
    )
    parser.add_argument(
        '-x', '--exclude', type=str, default='',
        help='comma separated list of test indexes to exclude e.g. 54,88888'
    )
    parser.add_argument(
        '-t', '--n_trials', type=int, default=1,
        help='Number of times to generate and evaluate'
    )
    parser.add_argument(
        '-d', '--device', type=str, default="cuda:0",
        help='Device to use for evaluation'
    )


    args = parser.parse_args()
    main(args)
