import argparse

from cdpo import pipelines
from cdpo.data_utils import (
    get_default_dir,
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
    parser.add_argument(
        '-d', '--device', type=str,
        help='Device to use for training'
    )

    args = parser.parse_args()

    pipelines.sft_and_dpo(args.params, args.output_dir, args.device)

