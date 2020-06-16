from Attention import Attention
from LSTM import LSTM
import torch
import torch.nn as nn

class BAN(nn.Module):
    def __init__(self, embedding_matrix, embed_dim, hidden_dim, polarities):
        super(BAN, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.context_lstm = LSTM(embed_dim, hidden_dim)
        self.aspect_lstm = LSTM(embed_dim, hidden_dim)
        self.aspect_attention = Attention(hidden_dim * 2)
        self.context_attention = Attention(hidden_dim * 2)
        self.dense = nn.Linear(hidden_dim * 2, polarities)

    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        text_raw_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)

        context = self.embed(text_raw_indices)
        aspect = self.embed(aspect_indices)
        context, (_, _) = self.context_lstm(context, text_raw_len)
        aspect, (_, _) = self.aspect_lstm(aspect, aspect_len)

        aspect_len = torch.tensor(aspect_len, dtype=torch.float)
        aspect_pool = torch.sum(aspect, dim=1)
        aspect_pool = torch.div(aspect_pool, aspect_len.view(aspect_len.size(0), 1))

        context_final, context_score = self.context_attention(context, aspect_pool)
        x = torch.squeeze(torch.bmm(context_score, context), dim=1)

        out = self.dense(x)

        return out
