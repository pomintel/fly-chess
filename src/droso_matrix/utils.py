import numpy as np


def normalize_matrix(W, mode=None):
    """
    Normalize a matrix using specified mode.

    Args:
        W (np.ndarray): Input matrix to normalize
        mode (str): Normalization mode ('minmax', 'clip', or None)

    Returns:
        np.ndarray: Normalized matrix
    """
    if mode is None:
        return W

    if mode == 'minmax':
        min_val = np.min(np.abs(W))
        max_val = np.max(np.abs(W))
        return (W - min_val) / (max_val - min_val)
    elif mode == 'clip':
        # Use 10th and 90th percentiles as clip range
        min_val = np.percentile(np.abs(W), 10)
        max_val = np.percentile(np.abs(W), 90)
        W = np.clip(W, min_val, max_val)
        return (W - min_val) / (max_val - min_val)  # Normalize to [0,1]
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def get_weight_matrix(base, mode):
    """
    Generate weight matrices based on different initialization modes.

    Args:
        base (np.ndarray): Base matrix to use as reference
        mode (str): Initialization mode. Options:
            - 'random': He initialization for ReLU
            - 'droso': Use base matrix as is
            - 'permuted': Randomly permute nonzero values
            - 'sparsity_matched': Random sparse matrix with same sparsity
            - 'row_permuted': Randomly permute rows of base matrix
            - 'col_permuted': Randomly permute columns of base matrix
            - 'eigenvalue_matched': Random matrix matching eigenvalue distribution
            - 'eigenvalue_permuted': Matrix with same eigenvalue distribution by permuting/discarding values
            - 'spectrum_sparsity_matched': Random matrix with same eigenvalue spectrum and sparsity level
            - 'low_rank_approximation': Low-rank approximation preserving 99% variance

    Returns:
        np.ndarray: Generated weight matrix
    """
    if mode == 'random':
        # use He Initialization for ReLU
        arr_np = (np.random.randn(*base.shape) / np.sqrt(base.shape[0])).astype(np.float32)
        return arr_np

    elif mode == 'droso':
        return base
    elif mode == 'droso-permute':
        temp = base.flatten()
        np.random.shuffle(temp)
        return temp.reshape(base.shape)

    elif mode == 'permuted':
        nonzero_vals = base[base != 0].astype(np.float32)
        np.random.shuffle(nonzero_vals)

        non_zero_count = len(nonzero_vals)
        idx = np.random.choice(base.size, non_zero_count, replace=False)
        arr_np = np.zeros_like(base, dtype=np.float32)

        arr_np_flat = arr_np.flatten()
        arr_np_flat[idx] = nonzero_vals
        arr_np = arr_np_flat.reshape(base.shape)

        return arr_np

    elif mode == 'sparsity_matched':
        non_zero = np.count_nonzero(base)
        mask = np.zeros(base.shape, dtype=np.float32)
        idx = np.random.permutation(mask.size)[:non_zero]
        mask.flat[idx] = 1
        scaling_factor = np.sqrt(non_zero / base.size)  # normalization factor
        arr_np = (np.random.randn(*base.shape) * scaling_factor).astype(np.float32) * mask
        return arr_np

    elif mode == 'row_permuted':
        # Randomly permute rows while preserving column structure
        row_indices = np.random.permutation(base.shape[0])
        # Make a copy to ensure contiguity in memory
        return base[row_indices, :].copy()

    elif mode == 'col_permuted':
        # Randomly permute columns while preserving row structure
        col_indices = np.random.permutation(base.shape[1])
        # Make a copy to ensure contiguity in memory
        return base[:, col_indices].copy()

    elif mode == 'eigenvalue_matched':
        # Get eigenvalue distribution of base matrix
        eigenvalues = np.linalg.eigvals(base)
        magnitudes = np.abs(eigenvalues)

        # Create random matrix with same shape
        random_matrix = np.random.randn(*base.shape)

        # Get its eigenvalues
        random_eigenvalues = np.linalg.eigvals(random_matrix)
        random_magnitudes = np.abs(random_eigenvalues)

        # Scale random eigenvalues to match base eigenvalue magnitudes
        scaling_factor = np.mean(magnitudes) / np.mean(random_magnitudes)
        scaled_eigenvalues = random_eigenvalues * scaling_factor

        # Reconstruct matrix with scaled eigenvalues
        # Note: This is an approximation since we can't perfectly reconstruct
        # the original matrix from just eigenvalues
        U, _ = np.linalg.qr(random_matrix)
        D = np.diag(scaled_eigenvalues)
        result = U @ D @ U.T
        return result.astype(np.float32).copy()  # Ensure contiguity

    elif mode == 'eigenvalue_permuted':
        # Get eigenvalue distribution of base matrix
        eigenvalues = np.linalg.eigvals(base)
        magnitudes = np.abs(eigenvalues)

        # Get the rank of the base matrix
        rank = np.linalg.matrix_rank(base)

        # Sort eigenvalues by magnitude
        sorted_indices = np.argsort(magnitudes)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]

        # Keep only the top 'rank' eigenvalues
        kept_eigenvalues = sorted_eigenvalues[:rank]

        # Create a diagonal matrix with the kept eigenvalues
        D = np.zeros_like(base, dtype=np.complex128)
        D[:rank, :rank] = np.diag(kept_eigenvalues)

        # Generate random orthogonal matrix
        Q = np.random.randn(*base.shape)
        Q, _ = np.linalg.qr(Q)

        # Reconstruct matrix
        reconstructed = Q @ D @ Q.T

        # Scale to match the magnitude distribution of the original matrix
        orig_magnitude = np.mean(np.abs(base))
        reconstructed_magnitude = np.mean(np.abs(reconstructed))
        scaling_factor = orig_magnitude / reconstructed_magnitude

        # Return a contiguous array
        return (reconstructed * scaling_factor).astype(np.float32).copy()

    elif mode == 'low_rank_approximation':
        # Perform SVD decomposition
        U, s, Vt = np.linalg.svd(base, full_matrices=False)

        # Compute cumulative variance
        explained_variance = np.cumsum(s ** 2) / np.sum(s ** 2)
        n_components = np.argmax(explained_variance >= 0.99) + 1

        # Reconstruct with top-k components
        U_k = U[:, :n_components]
        S_k = np.diag(s[:n_components])
        Vt_k = Vt[:n_components, :]
        low_rank_matrix = U_k @ S_k @ Vt_k

        return low_rank_matrix.astype(np.float32).copy()

    elif mode == 'random_structure_same_spectrum':
        # First compute the number of components needed
        U, s, Vt = np.linalg.svd(base, full_matrices=False)
        explained_variance = np.cumsum(s ** 2) / np.sum(s ** 2)
        n_components = np.argmax(explained_variance >= 0.99) + 1
        S_k = np.diag(s[:n_components])
        n = base.shape[0]

        # Generate random orthogonal structure
        U_rand, _ = np.linalg.qr(np.random.randn(n, n))
        V_rand, _ = np.linalg.qr(np.random.randn(n, n))
        random_structure_matrix = U_rand[:, :n_components] @ S_k @ V_rand[:, :n_components].T

        return random_structure_matrix.astype(np.float32).copy()

    else:
        raise ValueError(f"Unknown mode: {mode}")
