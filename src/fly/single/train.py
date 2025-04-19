import os
import pickle
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
import torch.nn.functional as F

from src.common import constants
from src.common.connectome_utils import load_connectivity_data
from src.common.matrix_utils import get_weight_matrix

from src.fly import config as config_lib
from src.fly.single.net import BasicCNN

# Add os.path for absolute path handling
import os.path


def get_out_path(droso_config):
    # Assume droso_config["result_path"] is already absolute
    out_folder_name = (
        f"{droso_config['exp_id']}_trial{1}_{droso_config.get('timesteps', 1)}Timesteps"
        + ("-signed" if droso_config.get("signed", True) else "")
    )
    # Use the absolute result_path directly
    return os.path.join(droso_config["result_path"], out_folder_name)


def train(
    train_config: config_lib.TrainConfig,
    test_config: config_lib.EvalConfig,
    build_data_loader: constants.DataLoaderBuilder,
    droso_config,
    device,
    trial_num=1,
):
    """Trains a predictor and returns the trained parameters."""
    torch.manual_seed(trial_num)
    np.random.seed(trial_num)
    train_config.data.seed = trial_num

    # Extract absolute data_root from config
    data_root = droso_config["data_root"]

    train_loader = build_data_loader(
        config=train_config.data, droso_config=droso_config, data_root=data_root
    )
    test_loader = build_data_loader(
        config=test_config.data, droso_config=droso_config, data_root=data_root
    )
    model = initialize_model(train_config, droso_config)
    model.to(device)

    criterion = soft_cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate)

    init_loss = eval_epoch(model, criterion, test_loader, device)
    print(
        f"[{droso_config['exp_id']}|{droso_config['timesteps']}Timesteps|Trial {trial_num}] Initial Test Loss: {init_loss:.4f}"
    )

    results = {
        "epoch_train_loss": [],
        "epoch_test_loss": [init_loss],
    }

    for epoch in range(train_config.num_epoch):
        train_loss = train_epoch(model, optimizer, criterion, train_loader, device)
        results["epoch_train_loss"].append(train_loss)

        test_loss = eval_epoch(model, criterion, test_loader, device)
        results["epoch_test_loss"].append(test_loss)

        print(
            f"[{droso_config['exp_id']}|Trial {trial_num}|Epoch {epoch + 1}] "
            f"Train Loss: {train_loss:.4f}, "
            f"Test Loss: {test_loss:.4f}"
        )

    # Save model and records
    out_path = get_out_path(droso_config)
    os.makedirs(out_path, exist_ok=True)

    # Paths are now absolute
    model_path = os.path.join(out_path, "model.pth")
    checkpoint = {"model": model, "config": droso_config}
    torch.save(checkpoint, model_path)

    # save loss records
    record_path = os.path.join(out_path, "record.pkl")
    with open(record_path, "wb") as f:
        pickle.dump(results, f)

    print(
        f"[{droso_config['exp_id']}|Trial {trial_num}] Done. Results saved to {out_path}"
    )
    return model, out_path


def train_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss, total = (
        0.0,
        0,
    )
    pbar = tqdm(train_loader, unit="batch", desc="Training")
    for data, target in pbar:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        output = model(data)
        if getattr(model, "pruning", False):
            ce_loss = criterion(output, target)
            l1_loss = (
                model.lambda_l1 * model.get_l1_loss()
                if model.lambda_l1 is not None
                else 0
            )
            loss = ce_loss + l1_loss
        else:
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        try:
            batch_size = data.size(0)
        except:
            batch_size = data[0].size(0)
        total_loss += loss.item() * batch_size
        total += batch_size

    return total_loss / total


def eval_epoch(model, criterion, test_loader, device):
    model.eval()
    with torch.no_grad():
        total_loss, total = (
            0.0,
            0,
        )
        pbar = tqdm(test_loader, unit="batch", desc="Eval", leave=True)
        for data, target in pbar:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            if getattr(model, "pruning", False):
                ce_loss = criterion(output, target)
                l1_loss = (
                    model.lambda_l1 * model.get_l1_loss()
                    if model.lambda_l1 is not None
                    else 0
                )
                loss = ce_loss + l1_loss
            else:
                loss = criterion(output, target)
            try:
                batch_size = data.size(0)
            except:
                batch_size = data[0].size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

        return total_loss / total


def soft_cross_entropy(logits, target):
    # logits: [N, C], target: [N, C] (both probabilities)
    log_probs = F.log_softmax(logits, dim=1)
    return -(target * log_probs).sum(dim=1).mean()


def initialize_model(config, config_data):
    data_setup = config_data.get("data")
    model_type = config.data.model_choice

    # Assume config_data contains absolute paths for csv_paths and annotation_path
    data_root = config_data[
        "data_root"
    ]  # Needed for potentially resolving paths within load_connectivity_data
    annotation_path = config_data["annotation_path"]
    signed_csv_path = config_data["csv_paths"]["signed"]

    if data_setup["data_choice"] == "chess_SV":
        num_out = config.data.num_return_buckets if data_setup["use_bucket"] else 2
    else:
        raise NotImplementedError("No other dataset is supported")

    conn = load_connectivity_data(
        # Pass absolute paths
        connectivity_path=signed_csv_path,
        annotation_path=annotation_path,
        rescale_factor=config_data.get("rescale_factor", 4e-2),
        sensory_type=config_data.get("sensory_type", "all"),
    )
    W_init = get_weight_matrix(conn["W"], config_data.get("init"))

    lora_config = config_data.get("lora", {})
    use_lora = lora_config.get("enabled", False)
    lora_rank = lora_config.get("rank", 8)
    lora_alpha = lora_config.get("alpha", 16)
    dropout_rate = config_data.get("dropout_rate", 0.2)

    if model_type == "basicCNN":
        return BasicCNN(
            W_init=W_init,
            sensory_dim=conn["W_ss"].shape[0],
            internal_dim=conn["W_rr"].shape[0],
            output_dim=conn["W_oo"].shape[0],
            num_out=num_out,
            trainable=config_data.get("trainable"),
            pruning=config_data.get("pruning"),
            target_nonzeros=np.count_nonzero(W_init),
            lambda_l1=config_data.get("lambda_l1"),
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            drop_out=config_data.get("drop_out", True),
            dropout_rate=dropout_rate,
            timesteps=config_data.get("timesteps"),
            filter_num=config_data.get("filter_num"),
            cumulate_output=config_data.get("cumulative", False),
            use_residual=config_data.get("residual", False),
            use_relu=config_data.get("use_relu", False),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
