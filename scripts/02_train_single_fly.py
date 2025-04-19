import copy
import os
import yaml
import torch
import os.path
from src.fly import config as config_lib
from src.fly import data_loader
from src.fly.single.train import train
from src.fly.evaluate import eval_DPU_on_puzzles


def main():
    # Determine project root (assuming script is in 'scripts' or run from root)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

    config_path = os.path.join(project_root, "configs", "config.yaml")
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Resolve paths to absolute paths relative to project_root
    def resolve_path(rel_path):
        return os.path.abspath(os.path.join(project_root, rel_path))

    base_config["data_root"] = resolve_path(base_config["data_root"])
    base_config["result_path"] = resolve_path(base_config["result_path"])
    base_config["annotation_path"] = resolve_path(base_config["annotation_path"])
    base_config["csv_paths"]["signed"] = resolve_path(
        base_config["csv_paths"]["signed"]
    )
    base_config["csv_paths"]["unsigned"] = resolve_path(
        base_config["csv_paths"]["unsigned"]
    )

    batch_size = base_config.pop("batch_size")
    num_epoch = base_config.pop("num_epoch")
    num_records = base_config.pop("train_num_sample")
    experiments = base_config.pop("experiments")
    # Store the timesteps from the base config
    global_timesteps = base_config.pop("timesteps")
    # Instead of modifying base_config directly, we'll set timesteps in each experiment's config
    for timesteps in [2, 5, 10, 15]:
        for exp_id in experiments.keys():
            droso_config = copy.deepcopy(base_config)
            droso_config["exp_id"] = exp_id + f"_{str(num_records)}"
            droso_config = {**droso_config, **experiments[exp_id]}
            # Set the timesteps explicitly in each experiment config
            droso_config["timesteps"] = timesteps

            if "filter_num" in droso_config:
                droso_config["exp_id"] = (
                    exp_id
                    + f"_{droso_config['filter_num']}filters"
                    + f"_{str(num_records)}"
                )

            # Check existence using absolute path (constructed inside train function now)
            # Need the result_path from droso_config for checking
            temp_out_folder_name = (
                f"{droso_config['exp_id']}_trial{1}_{droso_config.get('timesteps', 1)}Timesteps"
                + ("-signed" if droso_config.get("signed", True) else "")
            )
            model_output_path = os.path.join(
                droso_config["result_path"], temp_out_folder_name, "model.pth"
            )
            if os.path.exists(model_output_path):
                continue

            model_choice = droso_config["model_choice"]
            policy = droso_config["policy"]

            # TODO not sure if we'd want to bin the probs, maybe try both -> not using it
            num_return_buckets = 128 if droso_config["data"]["use_bucket"] else None

            train_config = config_lib.TrainConfig(
                learning_rate=0.0003,  # 1e-4,
                num_epoch=num_epoch,
                data=config_lib.DataConfig(
                    model_choice=model_choice,
                    batch_size=batch_size,
                    shuffle=True,
                    worker_count=0,  # 0 disables multiprocessing.
                    num_return_buckets=num_return_buckets,
                    policy=policy,
                    split="train",
                    num_records=num_records,  # total is 530310443 for SV
                ),
            )
            test_config = config_lib.EvalConfig(
                data=config_lib.DataConfig(
                    model_choice=model_choice,
                    batch_size=batch_size,
                    shuffle=False,
                    worker_count=0,  # 0 disables multiprocessing.
                    num_return_buckets=num_return_buckets,
                    policy=policy,  # pytype: disable=wrong-arg-types
                    split="test",
                    num_records=num_records // 10,
                ),
            )

            if torch.cuda.is_available():
                device = torch.device("cuda:1")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            print(f"Using device: {device}")

            model, out_path = train(
                train_config=train_config,
                test_config=test_config,
                build_data_loader=data_loader.build_data_loader,
                droso_config=droso_config,
                device=device,
            )

            eval_DPU_on_puzzles(out_path)
            # eval_DPU_on_puzzles(out_path,score_filter=400)


if __name__ == "__main__":
    # out_path = "results/DPU_CNN_Unlearnable_1filters_500000_trial1_2Timesteps-signed"
    # eval_DPU_on_puzzles(out_path)
    main()
