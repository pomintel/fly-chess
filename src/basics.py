import pandas as pd
import torch


def get_input_output_list():
    info_df = pd.read_pickle('Data_inspect/neuron_category.pkl')
    sensory_df = info_df[info_df['Category'] == 'sensory']
    input_list = ['all', 'ascending', 'sensory'] + sorted(list(set(sensory_df['sub_Category'].to_list())))
    output_list = ['all', 'output', 'DN-SEZ', 'DN-VNC', 'RGN']
    return input_list, output_list

def get_ct(name):
    info_df = pd.read_pickle('Data_inspect/neuron_category.pkl')
    if name == 'all':
        return len(info_df)
    elif name == 'output':
        output_list = {'DN-SEZ', 'DN-VNC', 'RGN'}
        filterd = info_df[info_df['Category'].isin(output_list)]
        return len(filterd)
    else:
        filterd = info_df[info_df['Category'] == name]
        if filterd.empty:
            filterd = info_df[info_df['sub_Category'] == name]
        return len(filterd)

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS device.")

    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    return device

def get_exp_name(config: dict):
    if config['init'] == 'droso':
        exp_id = 'DPU'
    else:
        raise NotImplementedError("Only support Droso init now")

    if config['learnable']:
        exp_id = exp_id + '_Learnable_' + config['learnable_type']
    else:
        exp_id = exp_id + '_Unlearnable'

    if config['model_choice'] == 'RNN':
        exp_id = exp_id + '_RNN'
    elif config['model_choice'] == 'CNN':
        exp_id = exp_id + '_CNN' + f"_{config['num_conv']}filters"
    exp_id = exp_id + f"_{config['num_heads']}heads"
    if config['residual']:
        exp_id = exp_id + '_Residual'

    exp_id = exp_id + f"_{str(config['timesteps'])}timesteps"
    exp_id = exp_id + f"_{str(config['train_num_sample'])}"
    exp_id = exp_id + f"_seed{str(config['seed'])}"

    return exp_id

