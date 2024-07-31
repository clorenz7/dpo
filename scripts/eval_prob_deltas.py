import argparse

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from cdpo import evaluation
from cdpo import data_utils


def main(args):
    device = "cuda:0"

    ds = data_utils.get_rlhf_data(n_test=2000)

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained(args.model)

    model.to(device)
    model.eval()
    with torch.no_grad():
        ds2 = evaluation.calculate_reference_probs(
            model, tokenizer,
            ds['test'],
            split_str="Assistant:", insert=True
        )

    try:
        ds2.save_to_disk(args.output_dir)
    except:
        import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate the model probability values for a response '
        'and add them to the dataset'
    )
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='Model name or directory path to load'
    )
    parser.add_argument(
        '-o', '--output_dir', type=str, required=True,
        help='Path to the output directory to save dataset'
    )
    args = parser.parse_args()

    main(args)
