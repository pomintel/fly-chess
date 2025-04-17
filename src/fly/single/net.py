import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicCNN(nn.Module):
    def __init__(
        self,
        W_init,
        sensory_dim: int,
        internal_dim: int,
        output_dim: int,
        num_out: int,
        trainable: bool = False,
        pruning: bool = False,
        target_nonzeros: int = None,
        lambda_l1: float = 1e-4,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16,
        drop_out=True,
        dropout_rate: float = 0.2,
        timesteps=5,
        filter_num=1,
        cumulate_output: bool = False,
        use_residual: bool = False,
        use_relu: bool = False,
    ):
        super().__init__()
        self.use_relu = use_relu
        self.filter_num = filter_num
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(8, self.filter_num, kernel_size=k, stride=1, padding=0)
                for k in range(1, 9)
            ]
        )
        self.cnn_output_dim = (
            sum((8 - k + 1) ** 2 for k in range(1, 9)) * self.filter_num
        )
        print(f"CNN output feature ct = {self.cnn_output_dim}")

        self.basicrnn = BasicRNN(
            W_init=W_init,
            input_dim=self.cnn_output_dim,
            sensory_dim=sensory_dim,
            internal_dim=internal_dim,
            output_dim=output_dim,
            num_out=num_out,
            trainable=trainable,
            pruning=pruning,
            target_nonzeros=target_nonzeros,
            lambda_l1=lambda_l1,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            drop_out=drop_out,
            dropout_rate=dropout_rate,
            timesteps=timesteps,
            cumulate_output=cumulate_output,
            use_residual=use_residual,
        )

    def forward(self, x):
        if hasattr(self, "use_relu") and self.use_relu:
            conv_outputs = [F.relu(conv(x.permute(0, 3, 1, 2))) for conv in self.convs]
        else:
            conv_outputs = [conv(x.permute(0, 3, 1, 2)) for conv in self.convs]

        conv_outputs = torch.cat(
            [out.reshape(out.size(0), -1) for out in conv_outputs], dim=1
        )

        return self.basicrnn(conv_outputs)


class BasicRNN(nn.Module):
    def __init__(
        self,
        W_init,
        input_dim: int,
        sensory_dim: int,
        internal_dim: int,
        output_dim: int,
        num_out: int,
        trainable: bool = False,
        pruning: bool = False,
        target_nonzeros: int = None,
        lambda_l1: float = 1e-4,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16,
        drop_out=True,
        dropout_rate: float = 0.2,
        cumulate_output: bool = False,
        use_residual: bool = False,
        timesteps: int = 5,
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
        """
        super().__init__()

        print(
            f"BasicRNN init: trainable={trainable}, pruning={pruning}, target_nonzeros={target_nonzeros}, lambda_l1={lambda_l1}"
        )
        print(f"LoRA config: use_lora={use_lora}, rank={lora_rank}, alpha={lora_alpha}")
        print(
            f"Regularization: dropout_rate={dropout_rate}" if drop_out else "No Dropout"
        )

        self.sensory_dim = sensory_dim
        self.internal_dim = internal_dim
        self.output_dim = output_dim
        self.total_dim = sensory_dim + internal_dim + output_dim

        self.pruning = pruning
        self.lambda_l1 = lambda_l1
        self.target_nonzeros = target_nonzeros

        self.cumulate_output = cumulate_output
        self.use_residual = use_residual
        if self.cumulate_output:
            assert self.use_residual == False

        print(
            f"W_init.shape: {W_init.shape}, sensory_dim: {sensory_dim}, internal_dim: {internal_dim}, output_dim: {output_dim}"
        )
        assert W_init.shape[0] == self.total_dim

        # Store initial weights and sparsity mask for similarity comparison
        self.register_buffer("W_init", torch.tensor(W_init, dtype=torch.float32))
        self.register_buffer(
            "sparsity_mask", torch.tensor(W_init != 0, dtype=torch.float32)
        )

        # Initialize base weight matrix (frozen DPU weights)
        W_init_tensor = torch.tensor(W_init, dtype=torch.float32)
        if trainable:
            self.W = nn.Parameter(W_init_tensor)
        else:
            self.register_buffer("W", W_init_tensor)

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
            self.lora_A.data.copy_(
                U[:, :lora_rank] * torch.sqrt(S[:lora_rank].unsqueeze(0))
            )
            # Initialize B with first r singular vectors
            self.lora_B.data.copy_(
                torch.sqrt(S[:lora_rank].unsqueeze(1)) * V[:, :lora_rank].t()
            )
            # Scale B to make initial LoRA contribution small
            self.lora_B.data.mul_(2.0)

        self.input_proj = nn.Linear(input_dim, sensory_dim)
        self.output_layer = nn.Linear(output_dim, num_out)
        self.activation = nn.ReLU()

        # Dropout layer for regularization
        self.use_dropout = drop_out
        self.dropout = nn.Dropout(dropout_rate)

        # set timesteps
        self.timesteps = timesteps

    def forward(self, x):
        """
        Forward pass with states S, I, O. We slice the effective W into sub-blocks:
          W_ss, W_sr, W_so, W_rs, W_rr, W_ro, W_os, W_or, W_oo
        """
        batch_size, device = x.shape[0], x.device

        # Just flatten the input
        x = x.view(batch_size, -1)

        # Get effective weight matrix (base + LoRA if enabled)
        W_eff = self.get_effective_W()

        # Partition the effective matrix W
        S, I, O = self.sensory_dim, self.internal_dim, self.output_dim
        W_ss = W_eff[0:S, 0:S]
        W_sr = W_eff[0:S, S : S + I]
        W_so = W_eff[0:S, S + I : S + I + O]
        W_rs = W_eff[S : S + I, 0:S]
        W_rr = W_eff[S : S + I, S : S + I]
        W_ro = W_eff[S : S + I, S + I : S + I + O]
        W_os = W_eff[S + I : S + I + O, 0:S]
        W_or = W_eff[S + I : S + I + O, S : S + I]
        W_oo = W_eff[S + I : S + I + O, S + I : S + I + O]

        # Initialize states S, I, O to zero
        S_state = torch.zeros(batch_size, S, device=device)
        I_state = torch.zeros(batch_size, I, device=device)
        O_state = torch.zeros(batch_size, O, device=device)

        E = self.input_proj(x)

        # patches for some previous saved models do not have this attribute
        # this part can be cleaned up a bit but not necessary....
        if hasattr(self, "use_dropout"):
            if self.use_dropout:
                E = self.dropout(E)
        else:
            E = self.dropout(E)
        if not hasattr(self, "cumulate_output"):
            self.cumulate_output = False
        if not hasattr(self, "use_residual"):
            self.use_residual = False

        if self.cumulate_output:
            cumulate_output = torch.zeros(batch_size, O, device=device)

        for t in range(self.timesteps):
            # single injection only
            E_t = E if (t == 0) else torch.zeros_like(E)

            S_next = S_state @ W_ss + E_t + I_state @ W_rs + O_state @ W_os
            I_next = I_state @ W_rr + S_state @ W_sr + O_state @ W_or
            O_next = O_state @ W_oo + I_state @ W_ro + S_state @ W_so
            if self.use_residual:
                S_next = S_state + S_next
                I_next = I_state + I_next
                O_next = O_state + O_next

            S_next = self.activation(S_next)
            I_next = self.activation(I_next)
            O_next = self.activation(O_next)

            S_state, I_state, O_state = S_next, I_next, O_next

            if self.cumulate_output:
                cumulate_output = O_state + cumulate_output

        if self.cumulate_output:
            out = self.output_layer(cumulate_output)
        else:
            out = self.output_layer(O_state)
        return out

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


class ThreeHiddenMLP(nn.Module):
    def __init__(
        self,
        input_size=784,
        hidden1_size=29,
        hidden2_size=147,
        hidden3_size=400,
        output_size=10,
        freeze=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.hidden3_size = hidden3_size
        self.output_size = output_size

        self.input_to_hidden1 = nn.Linear(input_size, hidden1_size, bias=True)

        if freeze:
            # If frozen, register as buffer
            self.register_buffer(
                "hidden1_to_hidden2", torch.randn(hidden1_size, hidden2_size)
            )
            self.register_buffer(
                "hidden2_to_hidden3", torch.randn(hidden2_size, hidden3_size)
            )
        else:
            # If trainable, register as parameter
            self.hidden1_to_hidden2 = nn.Parameter(
                torch.randn(hidden1_size, hidden2_size)
            )
            self.hidden2_to_hidden3 = nn.Parameter(
                torch.randn(hidden2_size, hidden3_size)
            )

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
    def __init__(
        self,
        input_size=784,
        hidden1_size=352,
        hidden2_size=352,
        output_size=10,
        freeze=False,
        use_weight_clipping=True,
    ):
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
            self.register_buffer(
                "hidden1_to_hidden2", torch.randn(hidden1_size, hidden2_size)
            )
        else:
            self.hidden1_to_hidden2 = nn.Parameter(
                torch.randn(hidden1_size, hidden2_size)
            )
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
        if not hasattr(self, "hidden1_to_hidden2") or not isinstance(
            self.hidden1_to_hidden2, nn.Parameter
        ):
            return

        with torch.no_grad():
            # Flatten weights for percentile calculation
            flat_weights = self.hidden1_to_hidden2.view(-1)

            # Calculate percentiles
            min_val = torch.quantile(flat_weights, self.percentile_clip_min / 100.0)
            max_val = torch.quantile(flat_weights, self.percentile_clip_max / 100.0)

            # Clip weights to the range [min_val, max_val]
            self.hidden1_to_hidden2.data.clamp_(min_val, max_val)

            print(
                f"Clipped weights to range [{min_val.item():.4f}, {max_val.item():.4f}] "
                f"({self.percentile_clip_min}% to {self.percentile_clip_max}%)"
            )

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
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=8, kernel_size=3, padding=1
        )
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
