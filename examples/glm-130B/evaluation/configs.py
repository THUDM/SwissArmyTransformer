from __future__ import annotations
from dataclass_wizard import YAMLWizard
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


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
    metrics: list[str] = field(default_factory=list)

    use_task_mask: bool = False
    unidirectional: bool = False
    max_seq_length: int = 2048
    source_file_pattern: str = "**/*.json*"


@dataclass
class MultiChoiceTaskConfig(BaseConfig):
    module = "evaluation.MultiChoiceTask"
    metrics: list[str] = field(default_factory=lambda: ["Accuracy"])


@dataclass
class GenerationTaskConfig(BaseConfig):
    module = "evaluation.GenerationTask"
    metrics: list[str] = field(default_factory=lambda: ["EM", "F1"])
    sampling_strategy: str = "BaseStrategy"
    num_beams: int = 1
    length_penalty: float = 0.0
    no_repeat_ngram_size: int = 0
    min_tgt_length: int = 0
