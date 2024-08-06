import argparse
import yaml

from datasets import (
    load_dataset,
    Dataset
)
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from cdpo import pipelines


def main(args):
    # Prepare the base dataset
    with open(args.params, 'r') as fp:
        train_params = yaml.safe_load(fp)

    # ----- Finetune basemodel (if necessary)
    ft_params = train_params.get('fine_tuning')
    if ft_params:
        # TODO:Evaluate the basemodel chosen vs reject

        # Do the finetuning with evaluation
        # TODO: Implement early stopping?
        # TODO: store plots in the output folder
        result, model, tokenizer = pipelines.fine_tuning(ft_params)
    else:
        model = tokenizer = None

    # ----- Do DPO
    dpo_result = pipelines.dpo_training_pipeline(
        train_params['dpo'], model, tokenizer
    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains and evaluates an LLM using DPO'
    )
    parser.add_argument(
        '-p', '--params', type=str, required=True,
        help='YAML file of training parameters'
    )
    # parser.add_argument(
    #     '-o', '--output_dir', type=str,
    #     help='Path to the output directory to save outputs'
    # )
    # parser.add_argument(
    #     '-g', '--gen_dir', type=str,
    #     help='Path to the generated dataset directory'
    # )

    args = parser.parse_args()

    # Read some JSON Parameters

    main(args)
