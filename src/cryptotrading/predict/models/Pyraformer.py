import torch.nn as nn
from cryptotrading.predict.layers.Pyraformer_EncDec import PyraformerEncoder


class Pyraformer(nn.Module):
    """
    Pyraformer: Pyramidal attention to reduce complexity

    window_size: list, the downsample window size in pyramidal attention.
    inner_size: int, the size of neighbour attention
    """

    def __init__(self, configs):
        super(Pyraformer, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout
        self.window_size = configs.window_size
        self.inner_size = configs.inner_size
        if self.window_size is None:
            self.window_size = [4, 4]

        self.encoder = PyraformerEncoder(
            configs.enc_in,
            self.seq_len,
            self.d_model,
            self.n_heads,
            self.e_layers,
            self.d_ff,
            self.dropout,
            self.window_size,
            self.inner_size
        )

        self.projection = nn.Linear(
            (len(self.window_size) + 1) * self.d_model,
            self.pred_len * configs.enc_in
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        enc_out = self.encoder(x_enc, x_mark_enc)[:, -1, :]
        dec_out = self.projection(enc_out).view(
            enc_out.size(0), self.pred_len, -1)

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
