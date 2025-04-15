import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle

class BasicRNN(nn.Module):
    def __init__(self, 
                 W_init,
                 input_dim: int,
                 sensory_dim: int,
                 internal_dim: int,
                 output_dim: int,
                 num_classes: int, 
                 sio: bool = True,
                 trainable: bool = False,
                 pruning: bool = False,
                 target_nonzeros: int = None,
                 lambda_l1: float = 1e-4,
                 use_lora: bool = False,
                 lora_rank: int = 8,
                 lora_alpha: float = 16,
                 dropout_rate: float = 0.2,
                 time_steps: int = 2,
                 use_output_projection: bool = True,
                 return_all_steps: bool = False,
                 ):
        """
        Unifies W_ss, W_sr, W_rs, W_rr, W_ro, W_or, W_so, W_oo, W_os into one
        big matrix W of shape (S+I+O, S+I+O). We'll slice it for sub-blocks.
        
        LoRA parameters:
        - use_lora: Whether to use LoRA adaptation
        - lora_rank: Rank of the LoRA matrices (r in the paper)
        - lora_alpha: Scaling factor for LoRA (alpha in the paper)
        
        Regularization parameters:
        - dropout_rate: Rate for dropout applied to the input layer
        
        Time steps:
        - time_steps: Number of time steps for RNN forward pass
        
        Output projection:
        - use_output_projection: Whether to project output to num_classes
        - return_all_steps: Whether to return outputs at all time steps
        """
        super().__init__()
        
        print(f"BasicRNN init: trainable={trainable}, pruning={pruning}, target_nonzeros={target_nonzeros}, lambda_l1={lambda_l1}")
        print(f"LoRA config: use_lora={use_lora}, rank={lora_rank}, alpha={lora_alpha}")
        print(f"Regularization: dropout_rate={dropout_rate}")
        print(f"Time steps: {time_steps}")
        print(f"Output projection: {use_output_projection}")
        print(f"Return all steps: {return_all_steps}")
        
        self.sensory_dim = sensory_dim
        self.internal_dim = internal_dim
        self.output_dim = output_dim
        self.total_dim = sensory_dim + internal_dim + output_dim
        self.sio = sio
        self.time_steps = time_steps
        self.use_output_projection = use_output_projection
        self.return_all_steps = return_all_steps
        
        self.pruning = pruning
        self.lambda_l1 = lambda_l1
        self.target_nonzeros = target_nonzeros
        
        print(f"W_init.shape: {W_init.shape}, sensory_dim: {sensory_dim}, internal_dim: {internal_dim}, output_dim: {output_dim}")
        assert W_init.shape[0] == self.total_dim
        
        # Store initial weights and sparsity mask for similarity comparison
        self.register_buffer('W_init', torch.tensor(W_init, dtype=torch.float32))
        self.register_buffer('sparsity_mask', torch.tensor(W_init != 0, dtype=torch.float32))
        
        # Initialize base weight matrix (frozen DPU weights)
        W_init_tensor = torch.tensor(W_init, dtype=torch.float32)
        if trainable:
            self.W = nn.Parameter(W_init_tensor)
        else:
            self.register_buffer('W', W_init_tensor)

        # LoRA parameters
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = lora_alpha / lora_rank

        if use_lora:
            # Initialize LoRA matrices A and B with improved initialization
            self.lora_A = nn.Parameter(torch.empty(self.total_dim, lora_rank))
            self.lora_B = nn.Parameter(torch.empty(lora_rank, self.total_dim))
            
            # Use SVD-based initialization for better starting point
            U, S, V = torch.svd(W_init_tensor)
            # Initialize A with first r singular vectors
            self.lora_A.data.copy_(U[:, :lora_rank] * torch.sqrt(S[:lora_rank].unsqueeze(0)))
            # Initialize B with first r singular vectors
            self.lora_B.data.copy_(torch.sqrt(S[:lora_rank].unsqueeze(1)) * V[:, :lora_rank].t())
            # Scale B to make initial LoRA contribution small
            self.lora_B.data.mul_(2.0)

        if self.sio:
            self.input_proj = nn.Linear(input_dim, sensory_dim)
            if use_output_projection:
                self.output_layer = nn.Linear(output_dim, num_classes)
        else:
            self.input_proj = nn.Linear(input_dim, self.total_dim)
            if use_output_projection:
                self.output_layer = nn.Linear(self.total_dim, num_classes)
            
        self.activation = nn.ReLU()
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)
    
    def get_effective_W(self):
        """Get the effective weight matrix including LoRA if enabled"""
        if not self.use_lora:
            return self.W
        
        # Compute LoRA contribution: (A × B) × scaling
        lora_contribution = (self.lora_A @ self.lora_B) * self.lora_scaling
        
        # Combine base weights with LoRA contribution
        effective_W = self.W + lora_contribution

        # Apply sparsity mask to maintain sparsity pattern
        return effective_W * self.sparsity_mask

    def calculate_matrix_similarity(self):
        """
        Calculate similarity metrics between initial and current effective weight matrix.
        """
        W_eff = self.get_effective_W()
        W_init = self.W_init
        
        # Cosine similarity
        W_eff_flat = W_eff.flatten()
        W_init_flat = W_init.flatten()
        cosine_sim = F.cosine_similarity(W_eff_flat.unsqueeze(0), W_init_flat.unsqueeze(0))
        
        # Frobenius norm of difference
        frob_diff = torch.norm(W_eff - W_init, p='fro')
        
        # Relative Frobenius norm
        rel_frob_diff = frob_diff / torch.norm(W_init, p='fro')
        
        # Sparsity comparison
        init_sparsity = (W_init == 0).float().mean()
        eff_sparsity = (W_eff == 0).float().mean()
        
        # Additional sparsity metrics
        shared_nonzeros = ((W_init != 0) & (W_eff != 0)).float().mean()
        new_nonzeros = ((W_init == 0) & (W_eff != 0)).float().mean()
        
        return {
            'cosine_similarity': cosine_sim.item(),
            'frobenius_diff': frob_diff.item(),
            'relative_frobenius_diff': rel_frob_diff.item(),
            'init_sparsity': init_sparsity.item(),
            'effective_sparsity': eff_sparsity.item(),
            'shared_nonzeros': shared_nonzeros.item(),
            'new_nonzeros': new_nonzeros.item()
        }

    def forward(self, x):
        """
        Forward pass with states S, I, O. We slice the effective W into sub-blocks:
          W_ss, W_sr, W_so, W_rs, W_rr, W_ro, W_os, W_or, W_oo
        """
        batch_size, device = x.shape[0], x.device
        
        # Use time_steps from config if not provided
        time_steps = self.time_steps if self.time_steps is not None else 2
        
        # Just flatten the input
        x = x.view(batch_size, -1)

        # Get effective weight matrix (base + LoRA if enabled)
        W_eff = self.get_effective_W()

        if self.sio:
            # Partition the effective matrix W
            S, I, O = self.sensory_dim, self.internal_dim, self.output_dim
            W_ss = W_eff[0:S,   0:S]
            W_sr = W_eff[0:S,   S:S+I]
            W_so = W_eff[0:S,   S+I:S+I+O]
            W_rs = W_eff[S:S+I, 0:S]
            W_rr = W_eff[S:S+I, S:S+I]
            W_ro = W_eff[S:S+I, S+I:S+I+O]
            W_os = W_eff[S+I:S+I+O, 0:S]
            W_or = W_eff[S+I:S+I+O, S:S+I]
            W_oo = W_eff[S+I:S+I+O, S+I:S+I+O]

            # Initialize states S, I, O to zero
            S_state = torch.zeros(batch_size, S, device=device)
            I_state = torch.zeros(batch_size, I, device=device)
            O_state = torch.zeros(batch_size, O, device=device)

            # Input projection with dropout
            E = self.dropout(self.input_proj(x))

            # If time_steps=1, ignore return_all_steps and just return the final output
            if time_steps == 1:
                # Single time step
                E_t = E
                S_next = self.activation(
                    S_state @ W_ss + E_t + I_state @ W_rs + O_state @ W_os
                )
                I_next = self.activation(
                    I_state @ W_rr + S_state @ W_sr + O_state @ W_or
                )
                O_next = self.activation(
                    O_state @ W_oo + I_state @ W_ro + S_state @ W_so
                )
                
                if self.use_output_projection:
                    return self.output_layer(O_next)
                return O_next

            # Store outputs at each time step if requested
            if self.return_all_steps:
                all_outputs = []

            for t in range(time_steps):
                E_t = E if (t % time_steps == 0) else torch.zeros_like(E)

                S_next = self.activation(
                    S_state @ W_ss + E_t + I_state @ W_rs + O_state @ W_os
                )
                I_next = self.activation(
                    I_state @ W_rr + S_state @ W_sr + O_state @ W_or
                )
                O_next = self.activation(
                    O_state @ W_oo + I_state @ W_ro + S_state @ W_so
                )

                S_state, I_state, O_state = S_next, I_next, O_next
                
                if self.return_all_steps:
                    if self.use_output_projection:
                        all_outputs.append(self.output_layer(O_state))
                    else:
                        all_outputs.append(O_state)
            
            if self.return_all_steps:
                return torch.stack(all_outputs, dim=1)  # [batch_size, time_steps, output_dim/num_classes]
            else:
                if self.use_output_projection:
                    return self.output_layer(O_state)
                return O_state
        else:
            # Initialize state to zero
            state = torch.zeros(batch_size, self.total_dim, device=device)
            
            # Input projection with dropout
            E = self.dropout(self.input_proj(x))
            
            # Store outputs at each time step if requested
            if self.return_all_steps:
                all_outputs = []
            
            for t in range(time_steps):
                E_t = E if (t % time_steps == 0) else torch.zeros_like(E)
                
                state_next = self.activation(
                    state @ W_eff + E_t
                )
                state = state_next
                
                if self.return_all_steps:
                    if self.use_output_projection:
                        all_outputs.append(self.output_layer(state))
                    else:
                        all_outputs.append(state)
            
            if self.return_all_steps:
                return torch.stack(all_outputs, dim=1)  # [batch_size, time_steps, output_dim/num_classes]
            else:
                if self.use_output_projection:
                    return self.output_layer(state)
                return state

    def get_l1_loss(self):
        """Compute L1 regularization loss on base weights only"""
        return self.W.abs().sum()

    def enforce_sparsity(self):
        """Hard threshold to maintain target sparsity level on base weights only"""
        with torch.no_grad():
            if self.target_nonzeros is None:
                return
                
            # Use reshape(-1) instead of view(-1) to handle non-contiguous tensors
            # This makes it work with column-permuted and other non-contiguous initializations
            W_flat = self.W.reshape(-1)
            numel = W_flat.numel()
            
            values, indices = torch.sort(W_flat.abs(), descending=True)
            if self.target_nonzeros >= numel:
                return
            threshold = values[self.target_nonzeros]
            
            # Zero out values below threshold
            mask = (self.W.abs() >= threshold)
            self.W.data.mul_(mask.float())

    def save_model(self, path, filename, metadata=None):
        """
        Save model and its configuration to a file.
        
        Parameters:
        -----------
        path : str
            Directory to save the model
        filename : str
            Base filename to use for saving
        metadata : dict, optional
            Additional metadata to save
        """
        # Create the directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), os.path.join(path, f'{filename}.pt'))
        
        # Save model configuration
        config = {
            'input_dim': self.input_proj.in_features,
            'sensory_dim': self.sensory_dim,
            'internal_dim': self.internal_dim,
            'output_dim': self.output_dim,
            'num_classes': self.output_layer.out_features,
            'W_init': self.W_init.cpu().numpy(),
            'trainable': isinstance(self.W, nn.Parameter),
            'pruning': self.pruning,
            'target_nonzeros': self.target_nonzeros,
            'lambda_l1': self.lambda_l1,
            'use_lora': self.use_lora,
            'lora_rank': getattr(self, 'lora_rank', 8),
            'lora_alpha': getattr(self, 'lora_alpha', 16),
            'sensory_type': getattr(self, 'sensory_type', 'visual'),
            'dropout_rate': getattr(self, 'dropout', nn.Dropout(0.2)).p,
            'use_position_encoding': getattr(self, 'use_position_encoding', False),
            'time_steps': self.time_steps,
            'use_output_projection': self.use_output_projection,
            'return_all_steps': self.return_all_steps
        }
        
        with open(os.path.join(path, 'model_config.pkl'), 'wb') as f:
            pickle.dump(config, f)
            
        # Save additional metadata if provided
        if metadata:
            with open(os.path.join(path, 'metadata.pkl'), 'wb') as f:
                pickle.dump(metadata, f)

    @classmethod
    def load_model(cls, path, device=None):
        """
        Load a saved model from a directory.
        """
        # Load model configuration
        with open(os.path.join(path, 'model_config.pkl'), 'rb') as f:
            config = pickle.load(f)
            
        # Update device if provided
        if device:
            config['device'] = device
            
        # Extract parameters
        input_dim = config.get('input_dim')
        sensory_dim = config.get('sensory_dim')
        internal_dim = config.get('internal_dim')
        output_dim = config.get('output_dim')
        num_classes = config.get('num_classes')
        W_init = config.get('W_init')
        trainable = config.get('trainable', False)
        pruning = config.get('pruning', False)
        target_nonzeros = config.get('target_nonzeros', None)
        lambda_l1 = config.get('lambda_l1', 1e-4)
        device = config.get('device', 'cpu')
        use_lora = config.get('use_lora', False)
        lora_rank = config.get('lora_rank', 8)
        lora_alpha = config.get('lora_alpha', 16)
        sensory_type = config.get('sensory_type', 'visual')
        dropout_rate = config.get('dropout_rate', 0.2)
        use_position_encoding = config.get('use_position_encoding', False)
        time_steps = config.get('time_steps', 5)
        use_output_projection = config.get('use_output_projection', True)
        return_all_steps = config.get('return_all_steps', False)
        
        # Create a new instance with the loaded parameters
        model = cls(
            W_init=W_init,
            input_dim=input_dim,
            sensory_dim=sensory_dim,
            internal_dim=internal_dim,
            output_dim=output_dim,
            num_classes=num_classes,
            trainable=trainable,
            pruning=pruning,
            target_nonzeros=target_nonzeros,
            lambda_l1=lambda_l1,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            sensory_type=sensory_type,
            dropout_rate=dropout_rate,
            use_position_encoding=use_position_encoding,
            time_steps=time_steps,
            use_output_projection=use_output_projection,
            return_all_steps=return_all_steps
        )
        
        # Load the model state
        model.load_state_dict(torch.load(os.path.join(path, 'model.pt'), map_location=device))
        model.to(device)
        
        # Load additional metadata if available
        metadata_path = os.path.join(path, 'metadata.pkl')
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                
        return model, metadata

class MultiSensoryRNN(nn.Module):
    def __init__(self, 
                 W_init_dict: dict,  # Dictionary mapping sensory type to W_init and dimensions
                 input_dim: int,
                 sensory_dims: dict,  # Dictionary mapping sensory type to dimension
                 num_classes: int,
                 sio: bool = True,
                 trainable: bool = False,
                 dropout_rate: float = 0.2,
                 time_steps: dict = None,  # Dictionary mapping sensory type to time steps
                 pooling_type: str = 'max',  # Options: 'max', 'mean', 'attention'
                 ):
        """
        Multi-sensory fusion RNN architecture with max pooling.
        
        Args:
            W_init_dict: Dictionary mapping sensory type to W_init matrix and dimensions
            input_dim: Input dimension for each sensory channel
            sensory_dims: Dictionary mapping sensory type to dimension
            num_classes: Number of output classes
            sio: Whether to use SIO architecture
            trainable: Whether RNN weights are trainable
            dropout_rate: Dropout rate
            time_steps: Dictionary mapping sensory type to number of time steps
            pooling_type: Type of pooling to use for temporal outputs ('max', 'mean', or 'attention')
        """
        super().__init__()
        
        self.sensory_dims = sensory_dims
        self.time_steps = time_steps or {sensory_type: 2 for sensory_type in sensory_dims.keys()}
        self.pooling_type = pooling_type
        
        # Initialize RNN modules for each sensory channel
        self.sensory_rnns = nn.ModuleDict()
        for sensory_type in sensory_dims.keys():
            # Get the pre-computed W_init and dimensions for this sensory type
            sensory_config = W_init_dict[sensory_type]
            W_init = sensory_config['W_init']
            sensory_dim = sensory_config['sensory_dim']  # Use actual sensory dimension from connectome
            internal_dim = sensory_config['internal_dim']
            output_dim = sensory_config['output_dim']
            
            self.sensory_rnns[sensory_type] = BasicRNN(
                W_init=W_init,
                input_dim=input_dim,
                sensory_dim=sensory_dim,  # Use actual sensory dimension
                internal_dim=internal_dim,
                output_dim=output_dim,
                num_classes=num_classes,
                sio=sio,
                trainable=trainable,
                pruning=False,
                dropout_rate=dropout_rate,
                time_steps=self.time_steps[sensory_type],
                use_output_projection=True,  # Enable output projection in individual RNNs
                return_all_steps=True  # Enable returning outputs at all time steps
            )
        
        # Attention mechanism for spatial attention (across sensory types)
        self.spatial_attention = nn.Sequential(
            nn.Linear(num_classes, num_classes),  # Query transformation
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_classes, 1)  # Attention scores
        )
        
        # Attention mechanism for temporal attention (across time steps)
        if pooling_type == 'attention':
            self.temporal_attention = nn.Sequential(
                nn.Linear(num_classes, num_classes),  # Query transformation
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(num_classes, 1)  # Attention scores
            )
    
    def get_sensory_output(self, x, sensory_type):
        """
        Process input through a sensory RNN and return output.
        
        Args:
            x: Input tensor
            sensory_type: Type of sensory input
        Returns:
            RNN output at all time steps [batch_size, time_steps, num_classes]
        """
        return self.sensory_rnns[sensory_type](x)
    
    def pool_temporal_outputs(self, temporal_outputs):
        """
        Pool outputs across time steps.
        
        Args:
            temporal_outputs: Tensor of shape [batch_size, time_steps, num_classes]
        Returns:
            Pooled output of shape [batch_size, num_classes]
        """
        if self.pooling_type == 'max':
            return torch.max(temporal_outputs, dim=1)[0]
        elif self.pooling_type == 'mean':
            return torch.mean(temporal_outputs, dim=1)
        elif self.pooling_type == 'attention':
            # Calculate attention scores for each time step
            attention_scores = self.temporal_attention(temporal_outputs)  # [batch_size, time_steps, 1]
            attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, time_steps, 1]
            # Apply attention weights
            return torch.sum(temporal_outputs * attention_weights, dim=1)  # [batch_size, num_classes]
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
    
    def forward(self, x_dict):
        """
        Forward pass through the multi-sensory network.
        
        Args:
            x_dict: Dictionary mapping sensory type to input tensor
        Returns:
            Classification logits
        """
        # Process each sensory input and get outputs at all time steps
        outputs = []
        for sensory_type, x in x_dict.items():
            temporal_outputs = self.get_sensory_output(x, sensory_type)
            pooled_output = self.pool_temporal_outputs(temporal_outputs)
            outputs.append(pooled_output)
        
        stacked_outputs = torch.stack(outputs, dim=1)  # [batch_size, num_sensory, num_classes]
        
        attention_scores = self.spatial_attention(stacked_outputs)  # [batch_size, num_sensory, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, num_sensory, 1]
        attended_output = torch.sum(stacked_outputs * attention_weights, dim=1)  # [batch_size, num_classes]
        
        return attended_output   


class ThreeHiddenMLP(nn.Module):
    def __init__(self, input_size=784, hidden1_size=29, hidden2_size=147, hidden3_size=400, output_size=10, 
                 freeze=False):
        super().__init__()
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.hidden3_size = hidden3_size
        self.output_size = output_size

        self.input_to_hidden1 = nn.Linear(input_size, hidden1_size, bias=True)

        if freeze:
            # If frozen, register as buffer
            self.register_buffer('hidden1_to_hidden2', torch.randn(hidden1_size, hidden2_size))
            self.register_buffer('hidden2_to_hidden3', torch.randn(hidden2_size, hidden3_size))
        else:
            # If trainable, register as parameter
            self.hidden1_to_hidden2 = nn.Parameter(torch.randn(hidden1_size, hidden2_size))
            self.hidden2_to_hidden3 = nn.Parameter(torch.randn(hidden2_size, hidden3_size))

        self.hidden3_to_output = nn.Linear(hidden3_size, output_size, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)

        hidden1 = self.relu(self.input_to_hidden1(x))
        hidden2 = self.relu(torch.matmul(hidden1, self.hidden1_to_hidden2))
        hidden3 = self.relu(torch.matmul(hidden2, self.hidden2_to_hidden3))

        output = self.hidden3_to_output(hidden3)
        return output
    
class TwoHiddenMLP(nn.Module):
    def __init__(self, input_size=784, hidden1_size=352, hidden2_size=352, output_size=10, 
                 freeze=False, use_weight_clipping=True):
        super().__init__()
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        # Add pruning attribute to avoid AttributeError
        self.pruning = False
        self.use_weight_clipping = use_weight_clipping
        self.percentile_clip_min = 10  # Bottom 10 percentile
        self.percentile_clip_max = 90  # Top 10 percentile

        self.input_to_hidden1 = nn.Linear(input_size, hidden1_size, bias=True)
        if freeze:
            self.register_buffer('hidden1_to_hidden2', torch.randn(hidden1_size, hidden2_size))
        else:
            self.hidden1_to_hidden2 = nn.Parameter(torch.randn(hidden1_size, hidden2_size))
        self.hidden2_to_output = nn.Linear(hidden2_size, output_size, bias=True)
        self.relu = nn.ReLU()
        
        # Initialize with clipping if enabled - only applied once during initialization
        if self.use_weight_clipping:
            self.clip_weights()
            print("Applied weight clipping during initialization only")

    def clip_weights(self):
        """
        Clip weights to lie between percentiles of their distribution.
        This is applied to hidden1_to_hidden2 weights only.
        """
        if not hasattr(self, 'hidden1_to_hidden2') or not isinstance(self.hidden1_to_hidden2, nn.Parameter):
            return
            
        with torch.no_grad():
            # Flatten weights for percentile calculation
            flat_weights = self.hidden1_to_hidden2.view(-1)
            
            # Calculate percentiles
            min_val = torch.quantile(flat_weights, self.percentile_clip_min / 100.0)
            max_val = torch.quantile(flat_weights, self.percentile_clip_max / 100.0)
            
            # Clip weights to the range [min_val, max_val]
            self.hidden1_to_hidden2.data.clamp_(min_val, max_val)
            
            print(f"Clipped weights to range [{min_val.item():.4f}, {max_val.item():.4f}] "
                  f"({self.percentile_clip_min}% to {self.percentile_clip_max}%)")

    def forward(self, x):
        x = x.view(x.size(0), -1)

        hidden1 = self.relu(self.input_to_hidden1(x))
        
        # No weight clipping during forward pass - removed from training process
        hidden2 = self.relu(torch.matmul(hidden1, self.hidden1_to_hidden2))
        output = self.hidden2_to_output(hidden2)
        return output   

class CNN(nn.Module):
    def __init__(self, input_channels=256, hidden_units=509, num_classes=10):
        super().__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 7 * 7, hidden_units)
        self.fc2 = nn.Linear(hidden_units, num_classes)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # First conv block
        x = self.pool1(self.relu(self.conv1(x)))
        
        # Second conv block
        x = self.pool2(self.relu(self.conv2(x)))
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x   