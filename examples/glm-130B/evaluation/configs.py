from __future__ import annotations
from dataclass_wizard import YAMLWizard
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict


class TaskType(Enum):
    MULTICHOICE = "mul"
    GENERATION = "gen"
    OTHER = "other"


@dataclass
class BaseConfig(YAMLWizard):
    name: str
    type: TaskType
    path: str

    module: Optional[str] = None
    metrics: List[str] = field(default_factory=list)

    use_task_mask: bool = False
    use_multitask_encoding: bool = False
    unidirectional: bool = False
    max_seq_length: int = 2048
    file_pattern: str | Dict[str, str] = "**/*.json*"

    micro_batch_size: int = 1

    def __post_init__(self):
        assert self.use_task_mask or not self.unidirectional, "[MASK] doesn't support unidirectional attention"


@dataclass
class MultiChoiceTaskConfig(BaseConfig):
    module = "evaluation.MultiChoiceTask"
    metrics: list[str] = field(default_factory=lambda: ["Accuracy"])


@dataclass
class GenerationTaskConfig(BaseConfig):
    module = "evaluation.GenerationTask"
    metrics: list[str] = field(default_factory=lambda: ["EM", "F1"])
    sampling_strategy: str = "BaseStrategy"
    num_beams: int = 4
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    min_gen_length: int = 0

    def __post_init__(self):
        assert self.micro_batch_size == 1, "Only support micro batch size = 1 for generation task"
