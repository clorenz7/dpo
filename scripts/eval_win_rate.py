import argparse

from datasets import (
    load_dataset,
    Dataset
)
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from cdpo import preference_eval
from cdpo import model_ops


def main(args):
    if args.gen_dir:
        ds2 = Dataset.load_from_disk(args.gen_dir)
    else:
        # Get the model
        device = "cuda:0"
        print("Loading the model...")
        tokenizer = model_ops.get_tokenizer("openai-community/gpt2")
        model = AutoModelForCausalLM.from_pretrained(args.model)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        model.to(device)

        # Load the dataset
        print("Loading the dataset...")
        n_samples = 100
        max_chars = 1280
        ds = load_dataset("Anthropic/hh-rlhf")
        ds['train'] = ds['train'].filter(lambda x: len(x['chosen']) <= max_chars)
        ds['test'] = ds['test'].filter(lambda x: len(x['chosen']) <= max_chars)
        ds['test'] = ds['test'].select(range(n_samples))

        print("Generating responses...")
        ds2 = preference_eval.generate_responses(
            model, tokenizer,
            ds['test'], device=device
        )

        if args.output_dir:
            ds2.save_to_disk(args.output_dir)

    print("Evaluating responses...")
    win_rate, fail_idxs, win_flags = preference_eval.evaluate_win_rate(ds2)
    print(f"Win Rate: {win_rate*100:0.1f}%")
    print(f"Fail indexes: {fail_idxs}")


    import ipdb; ipdb.set_trace()




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

    args = parser.parse_args()
    main(args)