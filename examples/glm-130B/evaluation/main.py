"""Evaluation"""

import os
import sys
from evaluate import main
from SwissArmyTransformer import get_args
import argparse
from SwissArmyTransformer.model import GLM130B
import torch

def add_evaluation_specific_args(parser):
    """Arguments for evaluation"""
    group = parser.add_argument_group('evaluation', 'evaluation Configurations')
    
    group.add_argument('--task', nargs='+', default=[], help="all task name under eval-data-path")

    group.add_argument('--micro-batch-size', type=int, help="micro batch size for evaluation")

    group.add_argument('--eval-data-path', type=str, help="evaluation dataset directory path")

    group.add_argument('--unified-multitask-encoding', action='store_true')

    return parser

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--sampling-strategy', type=str, default='BaseStrategy', help='type name of sampling strategy')
    GLM130B.add_model_specific_args(py_parser)
    add_evaluation_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)

    args = argparse.Namespace(**vars(args), **vars(known))
    
    with torch.no_grad():
        main(args)
