import torch
import torch.nn as nn
from LSTM import LSTM
from Attention import Attention
from SqueezeEmbedding import SqueezeEmbedding

class ATAE_LSTM(nn.Module):
    def __init__(self, embedding_matrix, embed_dim, hidden_dim, polarities_dim):
        super(ATAE_LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = LSTM(embed_dim*2, hidden_dim)
        self.attention = Attention(hidden_dim + embed_dim)
        self.dense = nn.Linear(hidden_dim, polarities_dim)

    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.tensor(torch.sum(aspect_indices != 0, dim=-1), dtype=torch.float)
        x = self.embed(text_raw_indices)
        x = self.squeeze_embedding(x, x_len)
        aspect = self.embed(aspect_indices)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.view(aspect_len.size(0), 1))
        aspect = torch.unsqueeze(aspect_pool, dim=1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)
        h, (_, _) = self.lstm(x, x_len)
        ha = torch.cat((h, aspect), dim=-1)
        _, score = self.attention(ha)
        output = torch.squeeze(torch.bmm(score, h), dim=1)
        out = self.dense(output)
        return out