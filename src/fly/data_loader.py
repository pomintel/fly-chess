import os
import os.path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.common import bagz, chess_utils
from src.common import constants
from src.fly.single import tokenizer
from src.fly import config as config_lib


def _process_fen(fen: str, config: dict) -> np.ndarray:
    return tokenizer.tokenize(fen)


def _process_move(move: str) -> np.ndarray:
    return np.asarray([chess_utils.MOVE_TO_ACTION[move]], dtype=np.int32)


def _process_win_prob(
    win_prob: float,
    return_buckets_edges: np.ndarray,
) -> np.ndarray:
    return chess_utils.compute_return_buckets_from_returns(
        returns=np.asarray([win_prob]),
        bins_edges=return_buckets_edges,
    )


def _get_uniform_bucket_edges(num_return_buckets: int):
    bucket_edges, bucket_values = chess_utils.get_uniform_buckets_edges_values(
        num_return_buckets
    )
    return bucket_edges, bucket_values


class BaseChessTransform:

    def __init__(self, config):
        self.config = {**config["data"], **{"model_choice": config["model_choice"]}}


class ConvertStateValueDataToSequence(BaseChessTransform):
    """Converts (fen, win_prob) into a sequence of tokens [S; R]."""

    def __init__(self, config):
        super().__init__(config)

    def __call__(self, fen: str, win_prob: float):
        state = _process_fen(fen, self.config)
        return state, np.array([win_prob, 1 - win_prob])


# TODO not implemented yet -> just keep it here. might need it for eval? - not really
# class ConvertActionValueDataToSequence(BaseChessTransform):
#     """Converts (fen, move, win_prob) into a sequence of tokens [S; A; R]."""
#     def __init__(self, num_return_buckets: int):
#         super().__init__(num_return_buckets=num_return_buckets)
#         # (s) + (a) + (r)
#         self._sequence_length = chess_tokenizer.SEQUENCE_LENGTH + 2
#
#     def __call__(self, fen: str, move: str, win_prob: float):
#         state = _process_fen(fen)
#         action = _process_move(move)
#         return_bucket = _process_win_prob(win_prob, self._return_buckets_edges)
#         sequence = np.concatenate([state, action, return_bucket])
#         return sequence, self.loss_mask


_TRANSFORMATION_BY_POLICY = {
    "state_value": ConvertStateValueDataToSequence,
}


class ChessDataset(Dataset):
    """
    PyTorch Dataset that reads from a .bag file, applies a transform,
    and returns (state, win_prob) after converting them to torch.Tensor.
    """

    def __init__(
        self,
        data_path: str,
        coder_name: str,
        transform_obj: BaseChessTransform,
        num_records: int = None,
        seed: int = 12345,
    ):
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


def build_data_loader(
    config: config_lib.DataConfig, droso_config: dict, data_root: str
) -> DataLoader:
    """
    Builds a DataLoader for chess data.

    Args:
        config: Data configuration object.
        droso_config: Drosophila-specific configuration.
        data_root: The absolute path to the root directory containing the data.
    """
    # Construct paths relative to the provided data_root
    base_chess_path = os.path.join(data_root, "chess")
    data_path = os.path.join(
        base_chess_path, f"{config.split}/{config.policy}_data.bag"
    )

    if config.split == "train":
        subsample_dir = os.path.join(base_chess_path, "subsample_train")
        subsample_data_path = os.path.join(
            subsample_dir,
            f"subsample_{config.num_records}_trial{config.seed}.bag",
        )
        # Check existence using the absolute path
        if os.path.exists(subsample_data_path):
            data_path = subsample_data_path
        # Optional: Create subsample directory if it doesn't exist, though this might belong elsewhere
        # else:
        #     os.makedirs(subsample_dir, exist_ok=True)

    if config.policy not in _TRANSFORMATION_BY_POLICY:
        raise ValueError(
            f"Unknown policy '{config.policy}' for model_choice '{config.model_choice}'."
        )
    transform_cls = _TRANSFORMATION_BY_POLICY[config.policy]
    transform_obj = transform_cls(droso_config)

    dataset = ChessDataset(
        data_path=data_path,
        coder_name=config.policy,
        transform_obj=transform_obj,
        num_records=config.num_records,
        seed=config.seed,
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.worker_count,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    return loader
