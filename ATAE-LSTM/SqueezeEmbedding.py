import torch
import torch.nn as nn

class SqueezeEmbedding(nn.Module):
    def __init__(self):
        super(SqueezeEmbedding, self).__init__()

    def forward(self, x, x_len):
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)
        out = torch.nn.utils.rnn.pad_packed_sequence(x_emb_p, batch_first=True)
        out = out[0]
        out = out[x_unsort_idx]
        return out