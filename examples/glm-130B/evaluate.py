import argparse
import importlib
import torch

from os.path import join, isdir, isfile, relpath
from glob import glob

from SwissArmyTransformer import get_args, get_tokenizer
from SwissArmyTransformer.arguments import initialize_distributed
from SwissArmyTransformer.training import load_checkpoint
from SwissArmyTransformer.model import GLM130B

from evaluation import BaseConfig, ModelForEvaluation, DEFAULT_CLASS, print_rank_0


def add_evaluation_specific_args(parser):
    """Arguments for evaluation"""
    group = parser.add_argument_group("evaluation", "Evaluation configurations")

    # Task
    group.add_argument("--task", nargs="+", default=[], help="All task config to evaluation")
    group.add_argument("--data-path", type=str, required=True, help="Data dir path for all tasks")
    return parser


def find_all_tasks(all_task_config_path):
    tasks = []
    for task in all_task_config_path:
        if isdir(task):
            tasks += [relpath(path, ".") for path in glob(join(task, "**/*.yaml"), recursive=True)]
        elif isfile(task):
            tasks.append(task)
    return tasks


def evaluate_all_tasks(data_path, model, tokenizer, all_task_config_path, task_classes):
    for config_path, task_class in zip(all_task_config_path, task_classes):
        config = task_class.config_class().from_yaml_file(config_path)
        config.path = join(data_path, config.path)
        task = task_class(model, tokenizer, config)
        task.evaluate()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    add_evaluation_specific_args(parser)
    GLM130B.add_model_specific_args(parser)
    known, args_list = parser.parse_known_args()
    sat_args = get_args(args_list)
    sat_args = argparse.Namespace(**vars(sat_args), **vars(known))
    sat_args.do_train = False
    initialize_distributed(sat_args)
    tokenizer = get_tokenizer(sat_args)

    sat_args.task = find_all_tasks(sat_args.task)

    task_classes = []
    for task_config_path in sat_args.task:
        config = BaseConfig.from_yaml_file(task_config_path)
        print_rank_0("> Loading task configs")
        if config.module:
            path = ".".join(config.module.split(".")[:-1])
            module = importlib.import_module(path)
            class_name = config.module.split(".")[-1]
            task_class = getattr(module, class_name)
            task_classes.append(task_class)
        else:
            task_classes.append(DEFAULT_CLASS[config.type])
        print_rank_0(f"  Task {config.name} loaded from config {task_config_path}")

    model = GLM130B(sat_args).half().to(sat_args.device)

    load_checkpoint(model, sat_args)
    torch.distributed.barrier()

    model = ModelForEvaluation(model)

    evaluate_all_tasks(sat_args.data_path, model, tokenizer, sat_args.task, task_classes)


if __name__ == "__main__":
    main()
