import torch
import time
import numpy as np
import torch.distributed as dist

from typing import Dict, Callable, Type, Tuple
from abc import ABC, abstractmethod
from glob import glob
from os.path import join, relpath
from collections import defaultdict

from SwissArmyTransformer.generation.sampling_strategies import BaseStrategy
from SwissArmyTransformer.tokenization.icetk_glm_130B.ice_tokenizer import _IceTokenizer

from .configs import BaseConfig, GenerationTaskConfig, MultiChoiceTaskConfig
from .model import ModelForEvaluation
from .dataset import ZeroShotDataset
from .utils import build_data_loader, gather_result, print_rank_0
from .strategies import DeterminedBeamSearchStrategy
from .metrics import qa_exact_match, qa_f1, accuracy_metric

DEFAULT_METRICS = {"EM": qa_exact_match, "F1": qa_f1, "Accuracy": accuracy_metric}


class BaseTask(ABC):
    model: ModelForEvaluation
    tokenizer: _IceTokenizer
    config: BaseConfig

    @classmethod
    def config_class(cls) -> Type[BaseConfig]:
        return BaseConfig

    @property
    def metrics(self) -> Dict[str, Callable]:
        return {metric: DEFAULT_METRICS[metric] for metric in self.config.metrics}

    def __init__(self, model: ModelForEvaluation, tokenizer: _IceTokenizer, config: BaseConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.config.metrics = list(self.metrics.keys())

        self.filelist = self.get_files()
        self.verbose = dist.get_rank() == 0

    def get_files(self):
        return [
            relpath(path, start=self.config.path)
            for path in sorted(glob(join(self.config.path, self.config.source_file_pattern), recursive=True))
        ]

    def build_dataset(self, file):
        return ZeroShotDataset(
            join(self.config.path, file),
            max_seq_length=self.config.max_seq_length,
            use_task_mask=self.config.use_task_mask,
            unidirectional=self.config.unidirectional,
        )

    def evaluate(self):
        dist.barrier()
        start = time.time()
        print_rank_0(self.config)
        print_rank_0(f"Evaluating task {self.config.name}")

        result_dict_all = {}

        for file in self.filelist:
            dataset = self.build_dataset(file)
            dataloader = build_data_loader(dataset, micro_batch_size=1, num_workers=1, drop_last=False)

            prediction = []
            with torch.no_grad():
                for _, batch in enumerate(dataloader):
                    prediction.append(self.predict_single_batch(batch))

            prediction = gather_result(prediction, len(dataset))
            result_dict = {key: metric(prediction, dataset.data) for key, metric in self.metrics.items()}
            result_dict_all[file] = (result_dict, len(dataset))

            if self.verbose:
                self.report_single_metrics(file, result_dict)

        print_rank_0(f"Finish task {self.config.name} in {time.time() - start}s:")

        if self.verbose:
            self.report_final_metrics(result_dict_all)

    def report_single_metrics(self, file: str, result_dict: Dict[str, float]):
        output_str = f"    Finish {file}"
        for key, value in result_dict.items():
            output_str += f", {key} = {value:.3f}%"
        print_rank_0(output_str)

    def report_final_metrics(self, result_dict_all: Dict[str, Tuple[Dict[str, float], int]]):
        metrics_dict = defaultdict(lambda: [])
        weight = []
        for file, (result_dict, length) in result_dict_all.items():
            for key, value in result_dict.item():
                metrics_dict[key].append(value)
            weight.append(length)
        for key, value in metrics_dict.items():
            idx = np.argmax(value)
            print_rank_0(
                f"    Metric {key}: max = {np.max(value):.3f}"
                f" | median = {np.median(value):.3f}, average = {(np.array(value) * np.array(weight) / np.sum(weight)).sum():.3f}"
                f" | ({'/'.join(metrics_dict.keys())}) = "
                f"{'/'.join(map(lambda x: f'{x[idx]:.3f}', metrics_dict.values()))}"
            )

    @abstractmethod
    def predict_single_batch(self, batch):
        pass


class GenerationTask(BaseTask, ABC):
    config: GenerationTaskConfig

    @classmethod
    def config_class(cls):
        return GenerationTaskConfig

    def __init__(self, model: ModelForEvaluation, tokenizer: _IceTokenizer, config_path: str):
        super(GenerationTask, self).__init__(model, tokenizer, config_path)

        end_tokens = [tokenizer.get_command("eop"), tokenizer.get_command("eos")]
        if self.config.sampling_strategy == "BaseStrategy":
            self.strategy = BaseStrategy(temperature=1.0, top_k=1, end_tokens=end_tokens)
        elif self.config.sampling_strategy == "BeamSearchStrategy":
            self.strategy = DeterminedBeamSearchStrategy(
                self.config.num_beams,
                length_penalty=self.config.length_penalty,
                consider_end=True,
                end_tokens=end_tokens,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                min_tgt_length=self.config.min_tgt_length,
            )
        else:
            raise ValueError(f"unknown strategy {self.config.sampling_strategy}")

    def predict_single_batch(self, batch):
        outputs = self.model.generate_text(batch, self.strategy, max_length=self.config.max_seq_length)
        return outputs


class MultiChoiceTask(BaseTask, ABC):
    config: MultiChoiceTaskConfig

    @classmethod
    def config_class(cls):
        return MultiChoiceTaskConfig

    def predict_single_batch(self, batch):
        return np.argmax(self.model.cond_log_prob(batch))
