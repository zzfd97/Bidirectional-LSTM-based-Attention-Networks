import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.RNN = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=False)

    def forward(self, x, x_len):
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)
        out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        ht = torch.transpose(ht, 0, 1)[x_unsort_idx]
        ht = torch.transpose(ht, 0, 1)
        out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=True)[0]
        out = out[x_unsort_idx]
        ct = torch.transpose(ct, 0, 1)[x_unsort_idx]
        ct = torch.transpose(ct, 0, 1)
        return out, (ht, ct)