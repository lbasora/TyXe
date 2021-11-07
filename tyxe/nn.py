import torch
import torch.nn.functional as F
from torch import nn


class LSTMFlipout(nn.LSTM):
    def forward(self, input, hx=None):
        return _forward(
            self.mode,
            input,
            hx,
            self._flat_weights,
            self.hidden_size,
            self.num_layers,
            self.bidirectional,
            self.batch_first,
            self.dropout,
        )


class GRUFlipout(nn.GRU):
    def forward(self, input, hx=None):
        return _forward(
            self.mode,
            input,
            hx,
            self._flat_weights,
            self.hidden_size,
            self.num_layers,
            self.bidirectional,
            self.batch_first,
            self.dropout,
        )


def to_rnn_flipout(rnn):
    if isinstance(rnn, nn.LSTM):
        rnn.__class__ = LSTMFlipout
    elif isinstance(rnn, nn.GRU):
        rnn.__class__ = LSTMFlipout


def _forward(
    mode, x, hx, w_b, hidden_size, num_layers, bidirectional, batch_first, dropout
):
    """Implementation based on:
    https://github.com/IntelLabs/bayesian-torch/blob/main/bayesian_torch/layers/flipout_layers/rnn_flipout.py
    """

    if mode not in ("LSTM", "GRU"):
        raise ValueError("Flipout only supported for LSTM/GRU RNN mode")

    if num_layers > 1:
        raise ValueError("Flipout for multilayer RNN not supported yet")

    if bidirectional:
        raise ValueError("Flipout for bidirectional RNN not supported yet")

    if dropout:
        raise ValueError("Flipout for RNN with dropout not supported yet")

    if hx is None:
        num_directions = 2 if bidirectional else 1
        max_batch_size = x.size(0) if batch_first else x.size(1)
        h_zeros = torch.zeros(
            num_layers * num_directions,
            max_batch_size,
            hidden_size,
            dtype=x.dtype,
            device=x.device,
        )
        if mode == "LSTM":
            c_zeros = torch.zeros(
                num_layers * num_directions,
                max_batch_size,
                hidden_size,
                dtype=x.dtype,
                device=x.device,
            )
            hx = (h_zeros, c_zeros)
        else:
            hx = h_zeros

    h_t, c_t = (
        (hx[0].squeeze(), hx[1].squeeze()) if len(hx) == 2 else (hx.squeeze(), None)
    )
    w = w_b[:2]
    b = w_b[2:] if len(w_b) == 4 else (None, None)
    w_ih, w_hh = w
    b_ih, b_hh = b
    output, c_ts = [], []
    seq_size = x.size(1) if batch_first else x.size(0)
    for t in range(seq_size):
        ih = F.linear(x[:, t, :], w_ih, b_ih)
        hh = F.linear(h_t, w_hh, b_hh)
        gates = ih + hh
        if mode == "LSTM":
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :hidden_size]),
                torch.sigmoid(gates[:, hidden_size : hidden_size * 2]),
                torch.tanh(gates[:, hidden_size * 2 : hidden_size * 3]),
                torch.sigmoid(gates[:, hidden_size * 3 :]),
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            c_ts.append(c_t.unsqueeze(0))
        elif mode == "GRU":
            r_t, z_t = (
                torch.sigmoid(gates[:, :hidden_size]),
                torch.sigmoid(gates[:, hidden_size : hidden_size * 2]),
            )
            n_t = torch.tanh(
                ih[:, hidden_size * 2 : hidden_size * 3]
                + r_t * hh[:, hidden_size * 2 : hidden_size * 3]
            )
            h_t = (1 - z_t) * n_t + z_t * h_t

        output.append(h_t.unsqueeze(0))

    output = torch.cat(output, dim=0)
    if batch_first:
        # reshape to (batch, sequence, feature)
        output = output.transpose(0, 1).contiguous()
    h_n = output[:, -1, :].unsqueeze(0)
    if mode == "LSTM":
        c_ts = torch.cat(c_ts, dim=0)
        if batch_first:
            # reshape to (batch, sequence, feature)
            c_ts = c_ts.transpose(0, 1).contiguous()
        c_n = c_ts[:, -1, :].unsqueeze(0)
        return output, h_n, c_n
    return output, h_n
