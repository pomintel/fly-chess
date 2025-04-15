import pandas as pd
import numpy as np

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

def load_connectivity_data(connectivity_path, annotation_path, rescale_factor=4e-2, sensory_type='all'):
    """
    Load and process the connectivity matrix and neuron annotations, splitting neurons into
    Sensory, Internal, and Output groups, then return a dictionary of 9 connectivity sub-matrices
    (as NumPy arrays) plus the neuron ID lists.
    
    Parameters:
    -----------
    connectivity_path : str
        Path to the connectivity matrix CSV file
    annotation_path : str
        Path to the neuron annotation CSV file
    rescale_factor : float
        Factor to rescale the connectivity weights
    normalization : str
        Type of normalization to apply ('minmax', 'clip', or None)
    sensory_type : str
        Type of sensory neurons to use as input. Can be:
        - 'olfactory': olfactory sensory neurons
        - 'visual': visual sensory neurons
        - 'gut': gut sensory neurons
        - 'respiratory': respiratory sensory neurons 
        - 'gustatory-external': external gustatory sensory neurons
        - 'all': all sensory neurons
    """

    df_annot = pd.read_csv(annotation_path)
    # output_types = {'DN-SEZ', 'DN-VNC', 'RGN'}
    output_types = {'DN-SEZ','RGN'}
    sensory_ids = []
    output_ids = []

    for _, row in df_annot.iterrows():
        cell_type = row['celltype']
        additional_annotations = row['additional_annotations']
        for col in ['left_id', 'right_id']:
            id_str = str(row[col]).lower()
            if id_str != "no pair":
                try:
                    neuron_id = int(id_str)
                except ValueError:
                    continue
                
                # Classify neuron as sensory based on selected type
                if cell_type == 'sensory':
                    if sensory_type == 'all':
                        sensory_ids.append(neuron_id)
                    elif sensory_type == 'visual' and additional_annotations == 'visual':
                        sensory_ids.append(neuron_id)
                    elif sensory_type == 'olfactory' and additional_annotations == 'olfactory':
                        sensory_ids.append(neuron_id)
                    elif sensory_type == 'gut' and additional_annotations == 'gut':
                        sensory_ids.append(neuron_id)
                    elif sensory_type == 'respiratory' and additional_annotations == 'respiratory':
                        sensory_ids.append(neuron_id)
                    elif sensory_type == 'gustatory-external' and additional_annotations == 'gustatory-external':
                        sensory_ids.append(neuron_id)
                
                # Classify output neurons
                elif cell_type in output_types:
                    output_ids.append(neuron_id)

    sensory_ids = sorted(set(sensory_ids))
    output_ids = sorted(set(output_ids))

    print(f"Annotation file: Found {len(sensory_ids)} {sensory_type} sensory neuron IDs")
    print(f"Annotation file: Found {len(output_ids)} output neuron IDs")

    df_conn = pd.read_csv(connectivity_path, index_col=0)

    # Apply normalization
    df_conn = df_conn * rescale_factor

    df_conn.index = df_conn.index.astype(int)
    df_conn.columns = df_conn.columns.astype(int)
    all_neuron_ids = sorted(df_conn.index.tolist())
    print(f"Connectivity matrix contains {len(all_neuron_ids)} neurons")

    valid_sensory_ids = [nid for nid in sensory_ids if nid in all_neuron_ids]
    valid_output_ids = [nid for nid in output_ids if nid in all_neuron_ids]

    # Define internal neurons as the rest ---
    valid_internal_ids = [
        nid for nid in all_neuron_ids
        if nid not in valid_sensory_ids and nid not in valid_output_ids
    ]

    print(f"After filtering, found {len(valid_sensory_ids)} {sensory_type} sensory neurons in matrix")
    print(f"After filtering, found {len(valid_output_ids)} output neurons in matrix")
    print(f"Remaining {len(valid_internal_ids)} neurons classified as internal")

    # Create the SIO-ordered adjacency matrix
    ordered_ids = valid_sensory_ids + valid_internal_ids + valid_output_ids
    adjacency = df_conn.loc[ordered_ids, ordered_ids].values  # shape: [N, N]
    
    # Calculate indices for each group in the SIO-ordered matrix
    num_sensory = len(valid_sensory_ids)
    num_internal = len(valid_internal_ids)
    num_output = len(valid_output_ids)

    W_ss = adjacency[:num_sensory, :num_sensory]
    W_sr = adjacency[:num_sensory, num_sensory:num_sensory+num_internal]
    W_so = adjacency[:num_sensory, num_sensory+num_internal:]

    W_rs = adjacency[num_sensory:num_sensory+num_internal, :num_sensory]
    W_rr = adjacency[num_sensory:num_sensory+num_internal, num_sensory:num_sensory+num_internal]
    W_ro = adjacency[num_sensory:num_sensory+num_internal, num_sensory+num_internal:]

    W_os = adjacency[num_sensory+num_internal:, :num_sensory]
    W_or = adjacency[num_sensory+num_internal:, num_sensory:num_sensory+num_internal]
    W_oo = adjacency[num_sensory+num_internal:, num_sensory+num_internal:]

    return {
        'W_original': df_conn.values,
        'W': adjacency,  # Now in SIO order
        'W_ss': W_ss,
        'W_sr': W_sr,
        'W_so': W_so,
        'W_rs': W_rs,
        'W_rr': W_rr,
        'W_ro': W_ro,
        'W_or': W_or,
        'W_os': W_os,
        'W_oo': W_oo,
        'sensory_ids': valid_sensory_ids,
        'internal_ids': valid_internal_ids,
        'output_ids': valid_output_ids,
        'sensory_type': sensory_type
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
