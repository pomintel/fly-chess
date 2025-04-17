from src.droso_matrix.connectome import load_connectivity_info
from src.droso_matrix.utils import get_weight_matrix
from src.model.net import *


def initialize_model(exp_config):
    if exp_config['data_choice'] == 'chess_SV': num_out = 2
    else: raise NotImplementedError("No other dataset is supported")

    droso_config = exp_config['droso_config']
    conn = load_connectivity_info(
        cfg_data=droso_config,
        input_type=exp_config.get('input_type', 'all'),
        output_type=exp_config.get('output_type', 'all'),
        sio=droso_config.get('sio', True)
    )
    W_init = get_weight_matrix(conn['W'], exp_config.get('init'))


    model_type = exp_config['model_choice']
    if model_type == 'CNN':
        return MultiHead_CNN(
            W_init=W_init,
            sensory_idx=conn['sensory_idx'],
            KC_idx=conn['KC_idx'],
            internal_idx=conn['internal_idx'],
            output_idx=conn['output_idx'],
            num_out=num_out,
            num_heads=exp_config.get('num_heads', 1),
            learnable=exp_config.get('learnable'),
            dropout_rate=exp_config.get('dropout_rate'),
            timesteps=exp_config.get('timesteps'),
            filter_num = exp_config.get('num_conv'),
            use_residual = exp_config.get('residual',False),
            learnable_type=exp_config.get('learnable_type', None),
        )

    if model_type == 'RNN':
        input_dim = (8 * 8 * 6 # piece location
                     + 4 # castling
                     + 16) # enpassant

        return MultiHead_RNN(
            W_init=W_init,
            input_dim=input_dim,
            sensory_idx=conn['sensory_idx'],
            KC_idx = conn['KC_idx'],
            internal_idx=conn['internal_idx'],
            output_idx=conn['output_idx'],
            num_out=num_out,
            num_heads = exp_config.get('num_heads',1),
            learnable=exp_config.get('learnable'),
            dropout_rate=exp_config.get('dropout_rate',0.2),
            timesteps=exp_config.get('timesteps'),
            use_residual=exp_config.get('residual'),
            learnable_type=exp_config.get('learnable_type', None),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
