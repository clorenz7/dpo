import argparse

from datasets import (
    load_dataset,
)
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from cdpo import evaluation
from cdpo import model_ops


def main(args):
    device = "cuda:0"

    ds = load_dataset("Anthropic/hh-rlhf")

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained(args.model)

    model.to(device)
    model.eval()
    with torch.no_grad():
        ds2 = evaluation.calculate_reference_probs(
            model, tokenizer,
            ds['test'].filter(lambda x: len(x['chosen']) <= 1280).select(range(2000)),
            split_str="Assistant:", insert=True
        )

    try:
        ds2.save_to_disk(args.output_dir, num_proc=3)
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