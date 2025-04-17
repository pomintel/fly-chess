import os
import numpy as np
import torch

from src.chess_utils import bagz
from src import constants
from config import chess_config as config_lib

def save_dataset_to_bag(dataset, output_bag_path):
    os.makedirs(os.path.dirname(output_bag_path), exist_ok=True)
    with bagz.BagWriter(output_bag_path) as writer:
        for idx in dataset.indices:
            raw_record = dataset.data_source[idx]
            writer.write(raw_record)

    print(f"Saved {len(dataset.indices)} records to {output_bag_path}")



def sample(
    train_config: config_lib.TrainConfig,
    build_data_loader: constants.DataLoaderBuilder,
    exp_config,
):
    seed = exp_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_config.data.seed = seed

    train_loader = build_data_loader(config=train_config.data,exp_config = exp_config)
    train_dataset = train_loader.dataset
    num_records = len(train_dataset)
    print(f"Dataset has {num_records} records.")
    out_train_set_path = os.path.join('data','chess_data','subsample_train',f'subsample_{num_records}_seed{seed}.bag')
    save_dataset_to_bag(train_dataset, out_train_set_path)