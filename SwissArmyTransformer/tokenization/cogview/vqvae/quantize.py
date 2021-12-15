import torch
from torch import nn
from torch import einsum
from torch.nn import functional as F

import distributed as dist_fn

class VectorQuantize(nn.Module):
    def __init__(self, 
                    hidden_dim,
                    embedding_dim,
                    n_embed,
                    commitment_cost=1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.commitment_cost = commitment_cost

        self.proj = nn.Conv2d(hidden_dim, embedding_dim, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)
        self.embed.weight.data.uniform_(-1. / n_embed, 1. / n_embed)

    def forward(self, z):
        B, C, H, W = z.shape

        z_e = self.proj(z)
        z_e = z_e.permute(0, 2, 3, 1) # (B, H, W, C)
        flatten = z_e.reshape(-1, self.embedding_dim)

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1, keepdim=True).t()
        )
        _, embed_ind = (-dist).max(1)
        embed_ind = embed_ind.view(B, H, W)

        z_q = self.embed_code(embed_ind)
        diff = self.commitment_cost * (z_q.detach() - z_e).pow(2).mean() \
                + (z_q - z_e.detach()).pow(2).mean()

        z_q = z_e + (z_q - z_e).detach()
        return z_q, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)
    

class VectorQuantizeEMA(nn.Module):
    def __init__(self,
                    hidden_dim,
                    embedding_dim,
                    n_embed,
                    commitment_cost=1,
                    decay=0.99,
                    eps=1e-5,
                    pre_proj=True,
                    training_loc=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.commitment_cost = commitment_cost
        self.training_loc = training_loc
        
        self.pre_proj = pre_proj
        if self.pre_proj:
            self.proj = nn.Conv2d(hidden_dim, embedding_dim, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)
        self.embed.weight.data.uniform_(-1. / n_embed, 1. / n_embed)
        
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", self.embed.weight.data.clone())
        
        self.decay = decay
        self.eps = eps
        
    def forward(self, z):
        B, C, H, W = z.shape
        
        if self.pre_proj:
            z_e = self.proj(z)
        else:
            z_e = z
        z_e = z_e.permute(0, 2, 3, 1) # (B, H, W, C)
        flatten = z_e.reshape(-1, self.embedding_dim)

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1, keepdim=True).t()
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(B, H, W)

        z_q = self.embed_code(embed_ind)
        
        if self.training_loc and self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = (flatten.transpose(0, 1) @ embed_onehot).transpose(0, 1)
            
            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)
            
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1-self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1-self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.weight.data.copy_(embed_normalized)
        
        diff = self.commitment_cost * (z_q.detach() - z_e).pow(2).mean()

        z_q = z_e + (z_q - z_e).detach()
        return z_q, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)
 

class GumbelQuantize(nn.Module):
    def __init__(self,
                    hidden_dim,
                    embedding_dim,
                    n_embed,
                    commitment_cost=1,
                    straight_through=True,
                    kl_weight=5e-4,
                    temp_init=1.,
                    eps=1e-5):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.commitment_cost = commitment_cost
        
        self.kl_weight = kl_weight
        self.temperature = temp_init
        self.eps = eps
        
        self.proj = nn.Conv2d(hidden_dim, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)
        self.embed.weight.data.uniform_(-1. / n_embed, 1. / n_embed)
        
        self.straight_through = straight_through
        
    def forward(self, z, temp=None):
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp
        
        B, C, H, W = z.shape
        
        z_e = self.proj(z)
        
        soft_one_hot = F.gumbel_softmax(z_e, tau=temp, dim=1, hard=hard)
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)
        
        qy = F.softmax(z_e, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + self.eps), dim=1).mean()
        
        embed_ind = soft_one_hot.argmax(dim=1)
        z_q = z_q.permute(0, 2, 3, 1)
        return z_q, diff, embed_ind
        
    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)
 