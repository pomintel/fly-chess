import os
import yaml
import argparse
from src.basics import get_exp_name
from config import chess_config as config_lib
from src.chess_utils import data_loader as data_loader
from src.train_utils import training
from src.evaluate_on_puzzles.evaluate_DPU_on_puzzles import eval_DPU_on_puzzles
from src.train_utils.training import get_out_path

with open("config/droso_config.yaml", "r") as f:
    DROSO_CONFIG = yaml.safe_load(f)

def main(exp_path: str):
    with open(os.path.join(exp_path,'exp_config.yaml'), "r") as f:
        base_config = yaml.safe_load(f)
    base_config['result_path'] = os.path.join(exp_path,base_config['result_path'])

    experiments = base_config.pop('experiments')
    for exp_id in experiments:
        exp_cfg = {**base_config, **experiments[exp_id]}
        exp_cfg['exp_id']     = get_exp_name(exp_cfg)
        exp_cfg['droso_config'] = DROSO_CONFIG

        out_path = get_out_path(exp_cfg)
        if os.path.exists(os.path.join(out_path, "model.pth")):
            continue

        train_config = config_lib.TrainConfig(
            learning_rate=exp_cfg['learning_rate'],
            num_epoch   =exp_cfg['num_epoch'],
            use_steps   =exp_cfg['use_steps'],
            num_steps   =exp_cfg['num_steps'],
            num_steps_eval=exp_cfg['num_steps_eval'],
            data=config_lib.DataConfig(
                model_choice=exp_cfg['model_choice'],
                batch_size  =exp_cfg['batch_size'],
                shuffle     =True,
                split       ='train',
                num_records =exp_cfg['train_num_sample']
            ),
        )
        test_config = config_lib.EvalConfig(
            data=config_lib.DataConfig(
                model_choice=exp_cfg['model_choice'],
                batch_size  =exp_cfg['batch_size'],
                shuffle     =False,
                split       ='test',
                num_records =exp_cfg['train_num_sample']//10,
            ),
        )

        model, out_path = training.train(
            train_config =train_config,
            test_config  =test_config,
            build_data_loader=data_loader.build_data_loader,
            exp_config   =exp_cfg,
        )

        eval_DPU_on_puzzles(out_path)
        # eval_DPU_on_puzzles(out_path, score_filter=500)



if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Run chess‐DPU experiments")
    p.add_argument(
        'exp_path',nargs='?', default='config', help="path to exp setup and save folder")
    args = p.parse_args()
    main(args.exp_path)