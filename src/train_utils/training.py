import os
import pickle
import numpy as np
from torch import optim

from src.model.init_model import initialize_model
from src.train_utils.losses import soft_cross_entropy
from src.train_utils.epoch_train_et_eval import *
from src.basics import get_device
from src import constants
from config import chess_config as config_lib


def get_out_path(config):
    return os.path.join(config['result_path'], config['exp_id'])


def train(
    train_config: config_lib.TrainConfig,
    test_config: config_lib.EvalConfig,
    build_data_loader: constants.DataLoaderBuilder,
    exp_config: dict
):
    """Trains a predictor and returns the trained parameters."""
    device = get_device()
    seed = exp_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_config.data.seed = seed

    train_loader = build_data_loader(config=train_config.data,exp_config = exp_config)
    test_loader = build_data_loader(config=test_config.data,exp_config = exp_config)

    model = initialize_model(exp_config)
    model.to(device)

    criterion = soft_cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate)
    init_loss = eval_epoch(model, criterion, test_loader, device)
    print(f"[{exp_config['exp_id']} | Initial Test Loss: {init_loss:.4f}")

    if train_config.use_steps:
      results = {
          "step_num":[],
          "step_train_loss": [],
          "step_test_loss": [],
          "init_test_loss": init_loss,
      }

      model.train()
      step, epoch = 1, 1
      data_iter = iter(train_loader)

      while step <= train_config.num_steps:
          interval_steps = min(train_config.num_steps_eval, train_config.num_steps - step + 1)
          train_loss, data_iter, step, epoch = train_steps(model, optimizer, criterion, train_loader, data_iter, device, interval_steps, step, epoch)

          results["step_num"].append(step)
          results["step_train_loss"].append(train_loss)

          test_loss = eval_epoch(model, criterion, test_loader, device)
          results["step_test_loss"].append(test_loss)

          print(
              f"[{exp_config['exp_id']}|Step {step}] "
              f"Train Loss: {train_loss:.4f}, "
              f"Test Loss: {test_loss:.4f}"
          )

    else: # train by epoch
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
              f"[{exp_config['exp_id']}|Epoch {epoch + 1}] "
              f"Train Loss: {train_loss:.4f}, "
              f"Test Loss: {test_loss:.4f}"
          )

    # Save model and records
    out_path = get_out_path(exp_config)
    os.makedirs(out_path, exist_ok=True)

    model_path = os.path.join(out_path, 'model.pth')
    checkpoint = {
      "model": model,
      "config": exp_config
    }
    torch.save(checkpoint, model_path)

    # save loss records
    with open(os.path.join(out_path, 'record.pkl'), "wb") as f:
      pickle.dump(results, f)

    print(f"[{exp_config['exp_id']} Done. Results saved to {out_path}")
    return model,out_path



