from LSTM import LSTM
import math
import torch
import torch.nn as nn
import torch.nn.functional as function

class AOA(nn.Module):
    def __init__(self, embedding_matrix, embed_dim, hidden_dim, polarities_dim):
        super(AOA, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.ctx_lstm = LSTM(embed_dim, hidden_dim)
        self.asp_lstm = LSTM(embed_dim, hidden_dim)
        self.dense = nn.Linear(2 * hidden_dim, polarities_dim)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        aspect_indices = inputs[1]
        ctx_len = torch.sum(text_raw_indices != 0, dim=1)
        asp_len = torch.sum(aspect_indices != 0, dim=1)
        ctx = self.embed(text_raw_indices)
        asp = self.embed(aspect_indices)
        ctx_out, (_, _) = self.ctx_lstm(ctx, ctx_len)
        asp_out, (_, _) = self.asp_lstm(asp, asp_len)
        interaction_mat = torch.matmul(ctx_out, torch.transpose(asp_out, 1, 2))
        alpha = function.softmax(interaction_mat, dim=1)
        beta = function.softmax(interaction_mat, dim=2)
        beta_avg = beta.mean(dim=1, keepdim=True)
        gamma = torch.matmul(alpha, beta_avg.transpose(1, 2))
        weighted_sum = torch.matmul(torch.transpose(ctx_out, 1, 2), gamma).squeeze(-1)
        out = self.dense(weighted_sum)
        return out

