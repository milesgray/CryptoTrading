import torch
import torch.nn as nn
import torch.nn.functional as F

class PyraformerConvLayer(nn.Module):
    """
    A Pyraformer convolutional layer.

    This layer applies a 1D convolution with a window size of `window_size` to the input tensor `x`. The stride is also
    set to `window_size`, which means that the output tensor will have a lower temporal resolution than the input. The
    output is then passed through batch normalization and the ELU activation function.

    Args:
        c_in (int): The number of input channels.
        window_size (int): The size of the sliding window for the convolution operation.

    Returns:
        Tensor: The output tensor of the layer.
    """

    def __init__(self, c_in, window_size):
        super(PyraformerConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=window_size,
                                  stride=window_size)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Bottleneck_Construct(nn.Module):
    """
    A bottleneck convolutional layer for the Pyraformer transformer.
    This layer implements a bottleneck convolutional CSCM (channel-split context modeling) operation.

    Args:
        d_model (int): The number of channels in the input and output tensors.
        window_size (int or list[int]): The size(s) of the sliding window(s) for the convolution operation(s).
            If an integer is provided, the same window size is used for all convolutional layers.
            If a list of integers is provided, each convolutional layer uses a different window size.
        d_inner (int): The dimensionality of the intermediate tensor produced by the first linear layer.

    Returns:
        Tensor: The output tensor of the layer.
    """

    def __init__(self, d_model, window_size, d_inner):
        super(Bottleneck_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([
                PyraformerConvLayer(d_inner, window_size),
                PyraformerConvLayer(d_inner, window_size),
                PyraformerConvLayer(d_inner, window_size)
            ])
        else:
            self.conv_layers = []
            for i in range(len(window_size)):
                self.conv_layers.append(PyraformerConvLayer(d_inner, window_size[i]))
            self.conv_layers = nn.ModuleList(self.conv_layers)
        self.up = Linear(d_inner, d_model)
        self.down = Linear(d_model, d_inner)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        temp_input = self.down(enc_input).permute(0, 2, 1)
        all_inputs = []
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)
            all_inputs.append(temp_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.up(all_inputs)
        all_inputs = torch.cat([enc_input, all_inputs], dim=1)

        all_inputs = self.norm(all_inputs)
        return all_inputs


class PositionwiseFeedForward(nn.Module):
    """
    A Two-layer position-wise feed-forward neural network for the Pyraformer transformer.

    Args:
        d_in (int): The number of input and output channels.
        d_hid (int): The number of channels in the hidden layer.
        dropout (float): The dropout probability.
        normalize_before (bool): If True, apply layer normalization before the feed-forward network; if False, apply
            layer normalization after the feed-forward network.

    Returns:
        Tensor: The output tensor of the layer.
    """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


class PyraformerEncoderLayer(nn.Module):
    """
    A single layer of the Pyraformer encoder.

    Args:
        d_model (int): The number of channels in the input and output tensors.
        d_inner (int): The number of channels in the intermediate tensor of the position-wise feed-forward network.
        n_head (int): The number of attention heads to use in the self-attention mechanism.
        dropout (float): The dropout probability.
        normalize_before (bool): If True, apply layer normalization before the self-attention mechanism; if False,
            apply layer normalization after the self-attention mechanism.

    Returns:
        Tensor: The output tensor of the layer.
    """

    def __init__(self, d_model, d_inner, n_head, dropout=0.1, normalize_before=True):
        super(PyraformerEncoderLayer, self).__init__()

        self.slf_attn = AttentionLayer(
            FullAttention(mask_flag=True, attention_dropout=dropout, output_attention=False),
            d_model, n_head)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, slf_attn_mask=None):
        attn_mask = RegularMask(slf_attn_mask)
        enc_output, _ = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class PyraformerEncoder(nn.Module):
    """
    An encoder model with a Pyraformer self-attention mechanism.

    Args:
        enc_in (int): The number of input channels.
        seq_len (int): The length of the input sequence.
        d_model (int): The number of channels in the input and output tensors.
        n_heads (int): The number of attention heads to use in the self-attention mechanism.
        e_layers (int): The number of Pyraformer encoder layers to use.
        d_ff (int): The number of channels in the intermediate tensor of the position-wise feed-forward network.
        dropout (float): The dropout probability.
        window_size (int or list[int]): The size(s) of the sliding window(s) for the bottleneck convolutional CSCM
            layer. If an integer is provided, the same window size is used for all convolutional layers. If a list of
            integers is provided, each convolutional layer uses a different window size.
        inner_size (int): The dimensionality of the intermediate tensor produced by the bottleneck linear layer.

    Returns:
        Tensor: The output tensor of the model.
    """

    def __init__(self, enc_in, seq_len, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, window_size=None, inner_size=4):
        super().__init__()

        if window_size is None:
            window_size = [4, 4]

        d_bottleneck = d_model // 4

        self.mask, self.all_size = get_mask_pam(seq_len, window_size, inner_size)

        self.indexes = refer_points(self.all_size, window_size)
        self.layers = nn.ModuleList([
            PyraformerEncoderLayer(d_model, d_ff, n_heads, dropout=dropout,
                                   normalize_before=False) for _ in range(e_layers)
        ])  # naive pyramid attention

        self.enc_embedding = DataEmbedding(
            enc_in, d_model, dropout)
        self.conv_layers = Bottleneck_Construct(
            d_model, window_size, d_bottleneck)

    def forward(self, x_enc, x_mark_enc):
        seq_enc = self.enc_embedding(x_enc, x_mark_enc)

        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)
        seq_enc = self.conv_layers(seq_enc)

        for i in range(len(self.layers)):
            seq_enc = self.layers[i](seq_enc, mask)

        indexes = self.indexes.repeat(seq_enc.size(
            0), 1, 1, seq_enc.size(2)).to(seq_enc.device)
        indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
        all_enc = torch.gather(seq_enc, 1, indexes)
        seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)

        return seq_enc
