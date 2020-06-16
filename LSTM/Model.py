import torch
import torch.nn as nn
from LSTM import LSTM


class Model(nn.Module):
    def __init__(self, embedding_matrix, embed_dim, hidden_dim, polarities_dim):
        super(Model, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = LSTM(embed_dim, hidden_dim)
        self.dense = nn.Linear(hidden_dim, polarities_dim)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        x = self.embed(text_raw_indices)
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        _, (h_n, _) = self.lstm(x, x_len)
        out = self.dense(h_n[0])
        return out