import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHead_RNN(nn.Module):
    def __init__(self,
                 W_init,
                 input_dim: int,
                 sensory_idx,
                 KC_idx,
                 internal_idx,
                 output_idx,
                 num_out: int,
                 num_heads: int,
                 learnable: bool = False,
                 dropout_rate: float = 0.2,
                 use_residual: bool = False,
                 timesteps: int = 5,
                 learnable_type = None
                 ):
        super().__init__()

        print(f"Regularization: dropout_rate={dropout_rate}")
        self.num_heads = num_heads
        self.output_layer = nn.Linear(len(output_idx) *  num_heads, num_out)
        self.heads = nn.ModuleList([
            BasicRNN(
                W_init,
                input_dim,
                sensory_idx,
                KC_idx,
                internal_idx,
                output_idx,
                learnable=learnable,
                dropout_rate=dropout_rate,
                use_residual=use_residual,
                timesteps=timesteps,
                learnable_type=learnable_type
            )
            for _ in range(num_heads)
        ])

    def forward(self, x):
        head_outs = [head(x) for head in self.heads]
        stacked = torch.concat(head_outs, dim=1)
        out = self.output_layer(stacked)
        return out




class BasicRNN(nn.Module):
    def __init__(self,
                 W_init,
                 input_dim: int,
                 sensory_idx,
                 KC_idx,
                 internal_idx,
                 output_idx,
                 learnable: bool = False,
                 dropout_rate: float = 0.2,
                 use_residual: bool = False,
                 timesteps: int = 5,
                 learnable_type = None
                 ):
        super().__init__()

        self.sensory_idx = sensory_idx
        self.KC_idx = KC_idx
        self.internal_idx = internal_idx
        self.output_idx = output_idx
        self.total_dim = len(self.sensory_idx) + len(self.KC_idx) + len(self.internal_idx) + len(self.output_idx)

        self.use_residual = use_residual

        W_init_tensor = torch.tensor(W_init, dtype=torch.float32)
        if learnable:
            mask = torch.zeros_like(W_init_tensor)
            if learnable_type == 'betweenKC':# train any connection that has one end in KC, changing weights = 3244
                non_zero_mask = torch.tensor(W_init != 0, dtype=torch.float32)
                mask[self.KC_idx, :] = non_zero_mask[self.KC_idx, :]
                mask[:, self.KC_idx] = non_zero_mask[:, self.KC_idx]
            elif learnable_type == 'withinKC': # train the whole 144 * 144, changing weights = 20736
                KC_idx_tensor = torch.tensor(self.KC_idx)
                rows, cols = torch.meshgrid(KC_idx_tensor, KC_idx_tensor, indexing='ij')
                mask[rows, cols] = 1.0
            elif learnable_type == 'all': # train all non-zero in the W
                mask = torch.tensor(W_init != 0, dtype=torch.float32)
            else:
                raise ValueError("Not supported learnable_type")
            self.register_buffer('mask', mask)

            self.W = nn.Parameter(W_init_tensor)
            self.W.register_hook(lambda grad: grad * self.mask)

        else:
            self.register_buffer('W', W_init_tensor)

        self.input_proj = nn.Linear(input_dim, len(self.sensory_idx))

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.timesteps = timesteps

    def forward(self, x):
        batch_size, device = x.shape[0], x.device
        x = x.view(batch_size, -1)

        E = self.dropout(self.input_proj(x))
        E = F.pad(E, (0, self.total_dim - E.size(1)), mode='constant', value=0)

        h_state = torch.zeros(batch_size, self.total_dim, device=device)

        for t in range(self.timesteps):
            E_t = E if (t == 0) else torch.zeros_like(E)
            h_next = h_state @ self.W + E_t
            if self.use_residual:
                h_next = h_state + h_next
            h_next = self.activation(h_next)
            h_state = h_next
        out = h_state[:, self.output_idx]
        return out






class MultiHead_CNN(nn.Module):
    def __init__(self,
                 W_init,
                 sensory_idx,
                 KC_idx,
                 internal_idx,
                 output_idx,
                 num_out: int,
                 num_heads: int,
                 learnable: bool = False,
                 dropout_rate: float = 0.2,
                 timesteps=5,
                 filter_num=1,
                 use_residual: bool = False,
                 learnable_type=None
                 ):
        super().__init__()
        print(f"Regularization: dropout_rate={dropout_rate}")
        self.num_heads = num_heads
        self.output_layer = nn.Linear(len(output_idx) * num_heads, num_out)
        self.heads = nn.ModuleList([
            BasicCNN(
            W_init=W_init,
            sensory_idx=sensory_idx,
            KC_idx=KC_idx,
            internal_idx=internal_idx,
            output_idx=output_idx,

            learnable=learnable,
            dropout_rate=dropout_rate,
            timesteps=timesteps,
            filter_num = filter_num,
            use_residual = use_residual,
            learnable_type=learnable_type,
        )
            for _ in range(num_heads)
        ])
    def forward(self, x):
        head_outs = [head(x) for head in self.heads]
        stacked = torch.concat(head_outs, dim=1)
        out = self.output_layer(stacked)
        return out




class BasicCNN(nn.Module):
    def __init__(self,
                 W_init,
                 sensory_idx,
                 KC_idx,
                 internal_idx,
                 output_idx,

                 learnable: bool = False,
                 dropout_rate: float = 0.2,
                 timesteps = 5,
                 filter_num = 1,
                 use_residual: bool = False,
                 learnable_type = None
                 ):
        super().__init__()
        self.filter_num = filter_num
        self.convs = nn.ModuleList([
            nn.Conv2d(8, self.filter_num, kernel_size=k, stride=1, padding=0)
            for k in range(1, 9)
        ])
        self.cnn_output_dim = sum((8 - k + 1) ** 2 for k in range(1, 9)) * self.filter_num
        print(f"CNN output feature ct = {self.cnn_output_dim}")

        self.basicrnn = BasicRNN(
            W_init=W_init,
            input_dim=self.cnn_output_dim,
            sensory_idx=sensory_idx,
            KC_idx=KC_idx,
            internal_idx=internal_idx,
            output_idx=output_idx,

            learnable=learnable,
            dropout_rate=dropout_rate,
            timesteps=timesteps,
            use_residual=use_residual,
            learnable_type = learnable_type,
        )

    def forward(self, x):
        conv_outputs = [F.relu(conv(x.permute(0, 3, 1, 2))) for conv in self.convs]
        conv_outputs = torch.cat([out.reshape(out.size(0), -1) for out in conv_outputs], dim=1)
        return self.basicrnn(conv_outputs)

