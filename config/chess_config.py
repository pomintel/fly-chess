import dataclasses
from typing import Literal


@dataclasses.dataclass(kw_only=True)
class DataConfig:
  model_choice: str
  batch_size: int
  shuffle: bool = False
  seed: int | None = 0
  split: Literal['train', 'test']
  num_records: int | None = None

@dataclasses.dataclass(kw_only=True)
class TrainConfig:
  num_epoch: int | None = 0
  use_steps: bool | None = False
  num_steps: int | None = 0
  num_steps_eval: int | None = 0
  data: DataConfig
  learning_rate: float

@dataclasses.dataclass(kw_only=True)
class EvalConfig:
  data: DataConfig
