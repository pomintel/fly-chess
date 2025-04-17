import copy
import yaml
from src.fly import config as config_lib
from src.fly import data_loader as data_loader
from src.fly import sampling


def main():
    with open("configs/config.yaml", "r") as f:
        base_config = yaml.safe_load(f)
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
