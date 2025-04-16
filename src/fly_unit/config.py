# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Defines the configuration dataclasses."""

import dataclasses
from typing import Literal

PolicyType = Literal['action_value', 'state_value', 'behavioral_cloning']

@dataclasses.dataclass(kw_only=True)
class DataConfig:
  """Config for the data generation."""
  model_choice: str
  # The batch size for the sequences.
  batch_size: int
  # Whether to shuffle the dataset (shuffling is applied per epoch).
  shuffle: bool = False
  # The seed used for shuffling and transformations of the data.
  seed: int | None = 0
  # Whether to drop partial batches.
  drop_remainder: bool = False
  # The number of child processes launched to parallelize the transformations.
  worker_count: int | None = 0
  # The number of return buckets.
  num_return_buckets: int
  # The dataset split.
  split: Literal['train', 'test']
  # The policy used to create the dataset.
  policy: PolicyType
  # The number of records to read from the dataset (can be useful when, e.g.,
  # the dataset does not fit into memory).
  num_records: int | None = None




@dataclasses.dataclass(kw_only=True)
class TrainConfig:
  """Config for the training function."""
  num_epoch: int
  # The data configuration for training.
  data: DataConfig
  # The learning rate for Adam.
  learning_rate: float


@dataclasses.dataclass(kw_only=True)
class EvalConfig:
  """Config for the evaluator."""
  # The data configuration for evaluation.
  data: DataConfig
