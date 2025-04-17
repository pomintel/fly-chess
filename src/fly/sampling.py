import os
import numpy as np
import torch

from src.common import bagz
from src.common import constants
from src.fly import config as config_lib


def get_out_path(droso_config):
    out_folder_name = (
        f"{droso_config['exp_id']}_trial{1}_{droso_config.get('timesteps', 1)}Timesteps"
        + ("-signed" if droso_config.get("signed", True) else "")
    )
    return os.path.join(droso_config["result_path"], out_folder_name)


def save_dataset_to_bag(dataset, output_bag_path):
    """
    Saves all records referenced by a ChessDataset to a new .bag file.
    The order of writing is exactly the order in `dataset.indices`.
    """
    os.makedirs(os.path.dirname(output_bag_path), exist_ok=True)
    with bagz.BagWriter(output_bag_path) as writer:
        for idx in dataset.indices:
            raw_record = dataset.data_source[idx]
            writer.write(raw_record)

    print(f"Saved {len(dataset.indices)} records to {output_bag_path}")


def sample(
    train_config: config_lib.TrainConfig,
    build_data_loader: constants.DataLoaderBuilder,
    droso_config,
    trial_num=1,
):
    torch.manual_seed(trial_num)
    np.random.seed(trial_num)
    train_config.data.seed = trial_num

    train_loader = build_data_loader(
        config=train_config.data, droso_config=droso_config
    )
    train_dataset = train_loader.dataset
    num_records = len(train_dataset)
    print(f"Dataset has {num_records} records.")
    out_train_set_path = os.path.join(
        "data",
        "chess",
        "subsample_train",
        f"subsample_{num_records}_trial{trial_num}.bag",
    )
    save_dataset_to_bag(train_dataset, out_train_set_path)
