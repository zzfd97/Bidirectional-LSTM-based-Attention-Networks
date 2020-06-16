import torch.nn as nn
import torch
import math

class Attention(nn.Module):
    def __init__(self, embed_dim):
        super(Attention, self).__init__()
        self.n_head = 1
        self.hidden_dim = embed_dim // self.n_head
        self.out_dim = embed_dim
        self.embed_dim = embed_dim
        self.w_k = nn.Linear(self.embed_dim, self.n_head * self.hidden_dim)
        self.w_q = nn.Linear(self.embed_dim, self.n_head * self.hidden_dim)
        self.proj = nn.Linear(self.n_head * self.hidden_dim, self.out_dim)
        self.dropout = nn.Dropout(0)
        self.weight = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        stdv = 1. / math.sqrt(self.hidden_dim)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]
        k_len = k.shape[1]
        q_len = q.shape[1]
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        qw = torch.matmul(qx, self.weight)
        kt = kx.permute(0, 2, 1)
        score = torch.bmm(qw, kt)
        score = nn.functional.softmax(score, dim=-1)
        output = torch.bmm(score, kx)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output, score