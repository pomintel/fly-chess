import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.chess_utils import bagz
from src import constants
from config import chess_config as config_lib
from src.chess_utils import tokenizer


def _process_fen(fen: str,config:dict) -> np.ndarray:
    return tokenizer.tokenize(fen, config)

class BaseChessTransform:

    def __init__(self, config):
        self.config = config

class ConvertStateValueDataToSequence(BaseChessTransform):
    """Converts (fen, win_prob) into a sequence of tokens [S; R]."""
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, fen: str, win_prob: float):
        state = _process_fen(fen,self.config)
        return state,np.array([win_prob,1-win_prob])


_TRANSFORMATION_BY_POLICY = {
    'state_value': ConvertStateValueDataToSequence,
}

class ChessDataset(Dataset):
    """
    PyTorch Dataset that reads from a .bag file, applies a transform,
    and returns (state, win_prob) after converting them to torch.Tensor.
    """
    def __init__(self,
                 data_path: str,
                 coder_name: str,
                 transform_obj: BaseChessTransform,
                 num_records: int = None,
                 seed: int = 12345):
        """
        Args:
          data_path: Path to the .bag file containing the raw chess data.
          coder_name: One of ['behavioral_cloning', 'state_value', 'action_value'],
                      used to decode each entry from the .bag file.
          transform_obj: A callable object (e.g., ConvertStateValueDataToSequence)
                         that takes raw fields and returns (state, win_prob).
          num_records: If provided, use a subsample of the data of this size.
          seed: Random seed used for reproducible subsampling.
        """
        super().__init__()
        self.data_source = bagz.BagDataSource(data_path)
        self.coder = constants.CODERS[coder_name]
        self.transform_obj = transform_obj

        total_records = len(self.data_source)
        if num_records is not None and num_records < total_records:
            rng = np.random.default_rng(seed)
            self.indices = rng.choice(total_records, size=num_records, replace=False)
            self.indices.sort()  # Ensure a deterministic order.
            self.num_records = num_records
        else:
            self.indices = np.arange(total_records)
            self.num_records = total_records

    def __len__(self):
        return self.num_records

    def __getitem__(self, idx: int):
        # Use subsampled indices for reproducibility.
        real_idx = self.indices[idx]
        raw_bytes = self.data_source[real_idx]
        decoded = self.coder.decode(raw_bytes)

        state, win_prob = self.transform_obj(*decoded)
        # The comment indicates:
        # - state is a nparray 8*8*x
        # - win_prob is a numpy array of shape (2,)

        state_tensor = torch.from_numpy(state).float()
        win_prob_tensor = torch.from_numpy(win_prob).float()
        return state_tensor, win_prob_tensor


def seed_worker(worker_id):
    """
    Seeds worker processes for reproducibility.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def build_data_loader(config: config_lib.DataConfig,exp_config: dict) -> DataLoader:
    if exp_config['data_choice'] == 'chess_SV':
        policy_name = 'state_value'
    else:
        raise NotImplementedError("Only chess_SV is supported")
    data_path = os.path.join(
        os.getcwd(),
        f'data/chess_data/{config.split}/{policy_name}_data.bag',
    )
    if config.split == 'train':
        subsample_data_path = os.path.join(
            os.getcwd(),
            f'data/chess_data/subsample_train/subsample_{config.num_records}_seed{config.seed}.bag',
        )
        if os.path.exists(subsample_data_path):
            data_path = subsample_data_path
    if policy_name not in _TRANSFORMATION_BY_POLICY:
        raise ValueError(f"Unknown policy '{policy_name}' for model_choice '{config.model_choice}'.")
    transform_cls = _TRANSFORMATION_BY_POLICY[policy_name]
    transform_obj = transform_cls(exp_config)

    dataset = ChessDataset(
        data_path=data_path,
        coder_name=policy_name,
        transform_obj=transform_obj,
        num_records=config.num_records,
        seed=config.seed
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator
    )

    return loader