import argparse
import os
import yaml

import torch

from cdpo import pipelines
from cdpo.data_utils import get_default_dir


def main(args):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Prepare the base dataset
    with open(args.params, 'r') as fp:
        train_params = yaml.safe_load(fp)

    exp_name = os.path.splitext(os.path.basename(args.params))[0]
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # ----- Finetune basemodel (if necessary)
    ft_params = train_params.get('fine_tuning')
    if ft_params:
        # TODO:Evaluate the basemodel chosen vs reject

        # Do the finetuning with evaluation
        # TODO: Implement early stopping?
        result, model, tokenizer = pipelines.fine_tuning(
            ft_params, output_dir, device=device
        )
    else:
        model = tokenizer = None

    # ----- Do DPO
    dpo_result = pipelines.dpo_training_pipeline(
        train_params['dpo'], output_dir,
        model=model, tokenizer=tokenizer,
        device=device
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains and evaluates an LLM using DPO'
    )
    parser.add_argument(
        '-p', '--params', type=str, required=True,
        help='YAML file of training parameters'
    )
    parser.add_argument(
        '-o', '--output_dir', type=str, default=get_default_dir(),
        help='Directory where experiment outputs are saved. '
             'Defaults to ~/cdpo_results/{base_file_name} or CDPO_DEFAULT_DIR env var'
    )

    args = parser.parse_args()
    main(args)
