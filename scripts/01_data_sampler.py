import copy
import yaml
import os
import os.path
from src.fly import config as config_lib
from src.fly import data_loader as data_loader
from src.fly import sampling


def main():
    # Determine project root (assuming script is in 'scripts' or run from root)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

    config_path = os.path.join(project_root, "configs", "config.yaml")
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Resolve paths to absolute paths relative to project_root
    def resolve_path(rel_path):
        # Handle potential None or empty paths gracefully if needed
        if not rel_path:
            return rel_path
        return os.path.abspath(os.path.join(project_root, rel_path))

    # Make sure all relevant paths from base_config are resolved
    if "data_root" in base_config:
        base_config["data_root"] = resolve_path(base_config["data_root"])
    if "result_path" in base_config:
        base_config["result_path"] = resolve_path(base_config["result_path"])
    if "annotation_path" in base_config:
        base_config["annotation_path"] = resolve_path(base_config["annotation_path"])
    if "csv_paths" in base_config and "signed" in base_config["csv_paths"]:
        base_config["csv_paths"]["signed"] = resolve_path(
            base_config["csv_paths"]["signed"]
        )
    if "csv_paths" in base_config and "unsigned" in base_config["csv_paths"]:
        base_config["csv_paths"]["unsigned"] = resolve_path(
            base_config["csv_paths"]["unsigned"]
        )

    batch_size = base_config.pop("batch_size")
    num_epoch = base_config.pop("num_epoch")
    num_records = base_config.pop("train_num_sample")
    experiments = base_config.pop("experiments")
    exp_id = list(experiments.keys())[0]
    droso_config = copy.deepcopy(base_config)
    droso_config["exp_id"] = exp_id + f"_{str(num_records)}"
    droso_config = {**droso_config, **experiments[exp_id]}
    if "filter_num" in droso_config:
        droso_config["exp_id"] = (
            exp_id + f"_{droso_config['filter_num']}filters" + f"_{str(num_records)}"
        )
    else:
        droso_config["exp_id"] = exp_id + f"_{str(num_records)}"

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

    sampling.sample(
        train_config=train_config,
        build_data_loader=data_loader.build_data_loader,
        droso_config=droso_config,
    )


if __name__ == "__main__":
    main()
