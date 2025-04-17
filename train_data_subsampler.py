import yaml
from config import chess_config as config_lib
from src.chess_utils import data_loader as data_loader, sampling


def main():
  with open("1%SV_1Msteps/exp_config.yaml", "r") as f:
      base_config = yaml.safe_load(f)
  _ = base_config.pop('experiments')

  num_records = base_config['train_num_sample']
  seed = base_config['seed']
  print(f"Sampling for {num_records} training samples with seed {seed}")

  train_config = config_lib.TrainConfig(
      learning_rate= 0.0, # placeholder
      data=config_lib.DataConfig(
          model_choice='NaN',
          batch_size=base_config['batch_size'],
          shuffle=True,
          split='train',
          num_records = num_records, # total is 530310443 for SV
      ),
  )

  sampling.sample(
      train_config=train_config,
      build_data_loader=data_loader.build_data_loader,
      exp_config = base_config,
  )

if __name__ == '__main__':
  main()
