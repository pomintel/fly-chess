import pandas as pd
import numpy as np

from src.droso_matrix.utils import normalize_matrix

def load_drosophila_matrix(csv_path, signed=False):
    """
    Load and process a Drosophila connectivity matrix.
    """
    W_df = pd.read_csv(csv_path, index_col=0, header=0)
    W = W_df.values.astype(np.float32)

    # Normalize depending on whether it's signed or unsigned
    if signed:
        max_abs = np.max(np.abs(W))
        W_norm = W / max_abs if max_abs != 0 else W
    else:
        W_min, W_max = W.min(), W.max()
        W_norm = (W - W_min) / (W_max - W_min + 1e-8)

    return W_norm

def load_connectivity_data(connectivity_path, annotation_path, rescale_factor=4e-2, normalization=None):
    """
    Load and preprocess connectivity matrix and annotation data for Drosophila.
    """
    df_annot = pd.read_csv(annotation_path)

    mask = (df_annot['celltype'] == 'sensory') & (df_annot['additional_annotations'] == 'visual')
    sensory_visual_ids = []
    for _, row in df_annot[mask].iterrows():
        for col in ['left_id', 'right_id']:
            id_str = str(row[col]).lower()
            if id_str != "no pair":
                sensory_visual_ids.append(int(id_str))

    sensory_visual_ids = sorted(set(sensory_visual_ids))
    print(f"Found {len(sensory_visual_ids)} sensory-visual neuron IDs")

    df_conn = pd.read_csv(connectivity_path, index_col=0)
    df_conn.index = df_conn.index.astype(int)
    df_conn.columns = df_conn.columns.astype(int)

    valid_sensory_ids = [nid for nid in sensory_visual_ids if nid in df_conn.index]
    other_ids = [nid for nid in df_conn.index if nid not in valid_sensory_ids]

    df_reindexed = df_conn.loc[valid_sensory_ids + other_ids, valid_sensory_ids + other_ids]

    adj_matrix = df_reindexed.values
    adj_matrix = normalize_matrix(adj_matrix, mode=normalization)
    adj_matrix = adj_matrix * rescale_factor

    num_S = len(valid_sensory_ids)
    return {
        'W': adj_matrix,
        'W_ss': adj_matrix[:num_S, :num_S],
        'W_sr': adj_matrix[:num_S, num_S:],
        'W_rs': adj_matrix[num_S:, :num_S],
        'W_rr': adj_matrix[num_S:, num_S:],
        'sensory_ids': valid_sensory_ids
    }


def load_sio_conn(connectivity_path, annotation_path, rescale_factor=4e-2, normalization='minmax',
                      input_type='visual',
                      output_type = 'output'):


    output_types = {'DN-SEZ', 'DN-VNC', 'RGN'}

    df_category = pd.read_pickle(annotation_path)
    if input_type in ['sensory', 'ascending']:
        df_input = df_category[df_category['Category'] == input_type]
    elif input_type != 'all':
        df_input = df_category[df_category['sub_Category'] == input_type]
    else:
        df_input = df_category
    input_ids = df_input['ID'].astype(int).tolist()

    if output_type == 'output':
        df_output = df_category[df_category['Category'].isin(output_types)]
    elif output_type != 'all':
        df_output = df_category[df_category['Category'] == output_type]
    else:
        df_output = df_category
    output_ids = df_output['ID'].astype(int).tolist()

    df_KC = df_category[df_category['Category'] == 'KC']
    KC_ids = df_KC['ID'].astype(int).tolist()

    KC_ids = sorted(set(KC_ids))
    input_ids = sorted(set(input_ids))
    output_ids = sorted(set(output_ids))

    df_conn = pd.read_csv(connectivity_path, index_col=0)
    df_conn.index = df_conn.index.astype(int)
    df_conn.columns = df_conn.columns.astype(int)
    all_neuron_ids = sorted(df_conn.index.tolist())
    print(f"Connectivity matrix contains {len(all_neuron_ids)} neurons")

    valid_sensory_ids = [nid for nid in input_ids if nid in all_neuron_ids]
    valid_output_ids = [nid for nid in output_ids if nid in all_neuron_ids]
    valid_KC_ids = [nid for nid in KC_ids if nid in all_neuron_ids]

    Other_ids = [
        nid for nid in all_neuron_ids
        if nid not in valid_sensory_ids and nid not in valid_output_ids and nid not in valid_KC_ids
    ]

    # Create the ordered adjacency matrix
    ordered_ids = valid_sensory_ids + valid_KC_ids + Other_ids + valid_output_ids
    df_conn_sio = df_conn.loc[ordered_ids, ordered_ids]
    adjacency = df_conn_sio.values  # shape: [N, N]

    # Apply normalization
    adjacency = normalize_matrix(adjacency, mode=normalization)
    adjacency = adjacency * rescale_factor

    return {
        'W': adjacency,  # Now in SIO order
        'sensory_idx': [ordered_ids.index(i) for i in valid_sensory_ids],
        'KC_idx': [ordered_ids.index(i) for i in valid_KC_ids],
        'internal_idx': [ordered_ids.index(i) for i in Other_ids],
        'output_idx': [ordered_ids.index(i) for i in valid_output_ids],
        'input_type': input_type,
        'output_type': output_type,
    }


def convert_unsigned_to_signed(sign_csv_path, adjacency_csv_path, out_csv_path):
    df = pd.read_csv(sign_csv_path)
    df["sign"] = df["sign"].apply(
        lambda x: "true" if str(x).strip().lower() == "true" else "false"
    )

    true_count = (df["sign"] == "true").sum()
    total_count = len(df)
    false_count = total_count - true_count
    true_pct = (true_count / total_count * 100) if total_count else 0
    false_pct = 100 - true_pct
    print(f"True sign count: {true_count} ({true_pct:.2f}%)")
    print(f"False sign count: {false_count} ({false_pct:.2f}%)")

    conn_matrix_df = pd.read_csv(adjacency_csv_path, index_col=0)
    conn_matrix_df.to_csv(out_csv_path)
    print(f"Saved signed connectivity matrix to: {out_csv_path}")


def load_connectivity_info(cfg_data, input_type, output_type, sio=True):
    if sio:
        return load_sio_conn(
            connectivity_path=cfg_data["csv_paths"]["signed"],
            annotation_path=cfg_data["annotation_path"],
            rescale_factor=cfg_data.get('rescale_factor', 4e-2),
            normalization=cfg_data.get('normalization', None),
            input_type=input_type,
            output_type=output_type,
        )
    else:
        return load_connectivity_data(
            connectivity_path=cfg_data["csv_paths"]["signed"],
            annotation_path=cfg_data["annotation_path"],
            rescale_factor=cfg_data.get('rescale_factor', 4e-2),
            normalization=cfg_data.get('normalization', None)
        )