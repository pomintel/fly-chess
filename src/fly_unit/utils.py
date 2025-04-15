import numpy as np

# Cache for SVD results
_svd_cache = {}

def get_svd_components(base, explained_variance):
    """
    Compute SVD components for a given matrix and cache the results.
    """
    cache_key = (id(base), explained_variance)
    if cache_key in _svd_cache:
        return _svd_cache[cache_key]
    
    U, s, Vt = np.linalg.svd(base, full_matrices=False)
    
    # For variance=1.0, we should use all non-zero singular values
    if explained_variance == 1.0:
        # Use machine epsilon to determine which singular values are effectively zero
        eps = np.finfo(s.dtype).eps
        n_components = np.sum(s > eps * s[0])
    else:
        explained_variance_ratio = np.cumsum(s**2) / np.sum(s**2)
        n_components = np.argmax(explained_variance_ratio >= explained_variance) + 1
    
    result = (U, s, Vt, n_components)
    _svd_cache[cache_key] = result
    return result

def get_weight_matrix(base, mode, explained_variance=0.1):
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
    
    elif mode == 'permuted':
        arr_np = base.copy()
        rows, cols = arr_np.shape
        # Create random permutations for rows and columns
        row_perm = np.random.permutation(rows)
        col_perm = np.random.permutation(cols)
        
        arr_np = arr_np[row_perm][:, col_perm]
        return arr_np.astype(np.float32)
        
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
    
    elif mode == 'same_eigenvalues_same_eigenvectors':
        # Get cached SVD components
        U, s, Vt, n_components = get_svd_components(base, explained_variance)
        print(f"Number of components: {n_components}")

        # For variance=1.0, we should return the original matrix
        if explained_variance == 1.0:
            return base.copy()

        # Reconstruct with top-k components
        U_k = U[:, :n_components]
        S_k = np.diag(s[:n_components])
        Vt_k = Vt[:n_components, :]
        low_rank_matrix = U_k @ S_k @ Vt_k

        return low_rank_matrix.astype(np.float32).copy()
    
    elif mode == 'same_eigenvalues_random_eigenvectors':
        # Get cached SVD components
        U, s, Vt, n_components = get_svd_components(base, explained_variance)
        print(f"Number of components: {n_components}")
        S_k = np.diag(s[:n_components])
        n = base.shape[0]

        # Generate random orthogonal structure
        U_rand, _ = np.linalg.qr(np.random.randn(n, n))
        V_rand, _ = np.linalg.qr(np.random.randn(n, n))
        random_structure_matrix = U_rand[:, :n_components] @ S_k @ V_rand[:, :n_components].T

        return random_structure_matrix.astype(np.float32).copy()
    
    elif mode == 'random_eigenvalues_same_eigenvectors':
        # Get cached SVD components
        U, s, Vt, n_components = get_svd_components(base, explained_variance)
        print(f"Number of components: {n_components}")
     
        # Keep original U and V, but randomize singular values
        # Generate random singular values with similar distribution
        s_mean = np.mean(s[:n_components])
        s_std = np.std(s[:n_components])
        random_s = np.random.normal(s_mean, s_std, n_components)
        random_s = np.clip(random_s, 0, None)  # Ensure non-negative
        
        # Sort random singular values in descending order
        random_s = np.sort(random_s)[::-1]
        
        # Create diagonal matrix with random singular values
        S_random = np.diag(random_s)
        
        # Reconstruct matrix with original U and V but random singular values
        random_singular_matrix = U[:, :n_components] @ S_random @ Vt[:n_components, :]
        
        return random_singular_matrix.astype(np.float32).copy()
    
    elif mode == 'random_eigenvalues_random_eigenvectors':
        # Get cached SVD components
        U, s, Vt, n_components = get_svd_components(base, explained_variance)
        print(f"Number of components: {n_components}")
        
        # Keep original U and V, but randomize singular values
        # Generate random singular values with similar distribution
        s_mean = np.mean(s[:n_components])
        s_std = np.std(s[:n_components])
        random_s = np.random.normal(s_mean, s_std, n_components)    
        
        # Sort random singular values in descending order
        random_s = np.sort(random_s)[::-1]
        
        # Create diagonal matrix with random singular values
        S_random = np.diag(random_s)
        n = base.shape[0]

        # Generate random orthogonal structure
        U_rand, _ = np.linalg.qr(np.random.randn(n, n))
        V_rand, _ = np.linalg.qr(np.random.randn(n, n))
        random_structure_matrix = U_rand[:, :n_components] @ S_random @ V_rand[:, :n_components].T

        return random_structure_matrix.astype(np.float32).copy()
    
    elif mode == 'identical_eigenvalues_same_eigenvectors':
        # Get cached SVD components
        U, s, Vt, n_components = get_svd_components(base, explained_variance)
        print(f"Number of components: {n_components}")

        # Create diagonal matrix with all ones for singular values
        S_ones = np.diag(np.ones(n_components))
        
        # Reconstruct matrix with original U and V but identical singular values
        identical_singular_matrix = U[:, :n_components] @ S_ones @ Vt[:n_components, :]
        
        return identical_singular_matrix.astype(np.float32).copy()
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
