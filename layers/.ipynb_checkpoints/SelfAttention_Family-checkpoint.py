import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import math
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
import os
from layers.Autoformer_EncDec import  my_Layernorm
from torch.nn.utils import weight_norm


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    

class AttentionGuidedProjection(nn.Module):
    def __init__(self, d_model, c_out):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads=4)
        self.projection = nn.Linear(d_model, c_out)
    
    def forward(self, x):
        # Assuming x is shape [seq_len, batch, features]
        attn_output, _ = self.attention(x, x, x)
        # Optionally apply pooling or select specific outputs
        pooled = attn_output.mean(dim=0)  # Example: mean pooling
        return self.projection(pooled)


















# *********** Best Model so far,  lowest loss so far,  Epoch : 12 Train Loss: 0.0057397 Vali Loss: 0.0376657
# class ProjectionAttentionWithFFT(nn.Module):
#     def __init__(self, d_model, n_heads, c_out, dropout_rate=0.1):
#         super(ProjectionAttentionWithFFT, self).__init__()

#         # Enhanced projection sequence with residual connections
#         self.projection = nn.Sequential(
#             nn.Linear(d_model, d_model//2),
#             nn.GELU(),
#             nn.LayerNorm(d_model//2),  # Changed from custom Layernorm for standardization
#             nn.Dropout(dropout_rate),
            
#             # Residual connection added
#             nn.Linear(d_model//2, d_model//4),
#             nn.GELU(),
#             nn.LayerNorm(d_model//4),
#             nn.Dropout(dropout_rate),

#             # Final projection
#             nn.Linear(d_model//4, c_out)
#         )
        
#         # Adding a residual connection that matches the input and output dimensions
#         self.residual = nn.Linear(d_model, c_out) if d_model != c_out else nn.Identity()

#     def forward(self, x):
#         # Applying the main projection path
#         proj_out = self.projection(x)
#         # Adding a skip connection
#         res_out = self.residual(x)
        
#         # Adding residual output to the projected output
#         out = proj_out + res_out  # Using residual connection to enhance training stability
        
#         return out












# ****** gave very good results about showed good convergece and went to 18 epocs with  5 patience 0.03 - 0.04

# class ProjectionAttentionWithFFT(nn.Module):
#     def __init__(self, d_model, n_heads, c_out, dropout_rate=0.1):
#         super(ProjectionAttentionWithFFT, self).__init__()

#         self.projection = nn.Sequential(
#             nn.Linear(d_model, d_model//2),
#             nn.GELU(),
#             nn.LayerNorm(d_model//2),
#             nn.Dropout(dropout_rate),
            
#             nn.Linear(d_model//2, d_model//4),
#             nn.GELU(),
#             nn.LayerNorm(d_model//4),
#             nn.Dropout(dropout_rate),

#             nn.Linear(d_model//4, c_out)
#         )
        
#         self.residual = nn.Linear(d_model, c_out) 

#         # Adding an LSTM layer before the final output for capturing temporal dependencies
#         self.lstm = nn.Sequential(
#             nn.Linear(d_model, d_model), 
#             nn.GELU(),
#             nn.LayerNorm(d_model),
#             nn.Dropout(dropout_rate),
#             nn.Linear(d_model, d_model//2), 
#             nn.GELU(),
#             nn.LayerNorm(d_model//2),
#             nn.Dropout(dropout_rate),
#             nn.Linear(d_model//2, d_model//4), 
#             nn.LayerNorm(d_model//4),
#             nn.Dropout(dropout_rate),
#             nn.LSTM(input_size=d_model//4, hidden_size=c_out, num_layers=1, batch_first=True))

#     def forward(self, x):

#         proj_out = self.projection(x)
#         res_out = self.residual(x)
#         out = proj_out + res_out
#         x, _ = self.lstm(x) 


#         return out+x










# **, gave quick covergence but stopped very early
# class ProjectionAttentionWithFFT(nn.Module):
#     def __init__(self, d_model, n_heads, c_out, dropout_rate=0.1):
#         super(ProjectionAttentionWithFFT, self).__init__()

#         self.projection = nn.Sequential(
#             nn.Linear(d_model, d_model//2),
#             nn.GELU(),
#             nn.LayerNorm(d_model//2),
#             nn.Dropout(dropout_rate),
            
#             nn.Linear(d_model//2, d_model//4),
#             nn.GELU(),
#             nn.LayerNorm(d_model//4),
#             nn.Dropout(dropout_rate),

#             nn.Linear(d_model//4, c_out)
#         )
        
#         self.residual = nn.Linear(d_model, c_out) 

#         # Adding an LSTM layer before the final output for capturing temporal dependencies
#         self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)

#     def forward(self, x):

#         x, _ = self.lstm(x) 
#         proj_out = self.projection(x)
#         res_out = self.residual(x)
#         out = proj_out + res_out

#         return out












# ***** gave very good results almost better than anyone
# class ProjectionAttentionWithFFT(nn.Module):
#     def __init__(self, d_model, n_heads, c_out, dropout_rate=0.1):
#         super(ProjectionAttentionWithFFT, self).__init__()

#         self.projection = nn.Sequential(
#             nn.Linear(d_model, d_model//2),
#             nn.GELU(),
#             nn.LayerNorm(d_model//2),
#             nn.Dropout(dropout_rate),
            
#             nn.Linear(d_model//2, d_model//4),
#             nn.GELU(),
#             nn.LayerNorm(d_model//4),
#             nn.Dropout(dropout_rate),

#             nn.Linear(d_model//4, c_out)
#         )
        
#         self.residual = nn.Linear(d_model, c_out) 

#         # Adding an LSTM layer before the final output for capturing temporal dependencies
#         self.lstm = nn.LSTM(input_size=d_model, hidden_size=c_out, num_layers=1, batch_first=True)

#     def forward(self, x):

#         proj_out = self.projection(x)
#         res_out = self.residual(x)
#         out = proj_out + res_out
#         x, _ = self.lstm(x) 


#         return out+x



# *** gave good results identified patters but didnt identified heights

# class ProjectionAttentionWithFFT(nn.Module):
#     def __init__(self, d_model, n_heads, c_out, dropout_rate=0.1):
#         super(ProjectionAttentionWithFFT, self).__init__()

#         self.projection = nn.Sequential(
#             nn.Linear(d_model, d_model//2),
#             nn.GELU(),
#             nn.LayerNorm(d_model//2),
#             nn.Dropout(dropout_rate),
            
#             nn.Linear(d_model//2, d_model//4),
#             nn.GELU(),
#             nn.LayerNorm(d_model//4),
#             nn.Dropout(dropout_rate),

#             nn.Linear(d_model//4, c_out)
#         )
        
#         self.residual = nn.Linear(d_model, c_out) 

#         # Adding an LSTM layer before the final output for capturing temporal dependencies
#         self.lstm = nn.Sequential(nn.Linear(d_model, d_model), nn.LSTM(input_size=d_model, hidden_size=c_out, num_layers=1, batch_first=True))

#     def forward(self, x):

#         proj_out = self.projection(x)
#         res_out = self.residual(x)
#         out = proj_out + res_out
#         x, _ = self.lstm(x) 


#         return out+x














# ****************  CAPTURED VERY GOOD PATTERN

# class ProjectionAttentionWithFFT(nn.Module):
#     def __init__(self, d_model, n_heads, c_out, dropout_rate=0.1):
#         super(ProjectionAttentionWithFFT, self).__init__()

#         # Enhanced convolutional layers with increased depth
#         self.conv_layers = nn.Sequential(
#             nn.Conv1d(in_channels=d_model, out_channels=d_model*2, kernel_size=3, padding=1),
#             nn.GELU(),
#             nn.BatchNorm1d(d_model*2),
#             nn.Dropout(dropout_rate),

#             nn.Conv1d(in_channels=d_model*2, out_channels=d_model, kernel_size=3, padding=1),
#             nn.GELU(),
#             nn.BatchNorm1d(d_model),
#             nn.Dropout(dropout_rate),

#             # Adding additional convolutional layers
#             nn.Conv1d(in_channels=d_model, out_channels=d_model//2, kernel_size=3, padding=1),
#             nn.GELU(),
#             nn.BatchNorm1d(d_model//2),
#             nn.Dropout(dropout_rate),

#             nn.Conv1d(in_channels=d_model//2, out_channels=d_model//4, kernel_size=3, padding=1),
#             nn.GELU(),
#             nn.BatchNorm1d(d_model//4),
#             nn.Dropout(dropout_rate)
#         )
        
#         self.projection = nn.Sequential(
#             nn.Linear(d_model//4, d_model // 8),
#             nn.GELU(),
#             my_Layernorm(d_model // 8),
#             nn.Dropout(dropout_rate),
            
#             nn.Linear(d_model // 8, c_out)
#         )
        
#         self.residual = nn.Linear(d_model//4, c_out) if d_model != c_out else nn.Identity()

#     def forward(self, x):
#         x = x.permute(0, 2, 1)  # Change to [batch, features, seq_len] for conv1d

#         x = self.conv_layers(x)
#         x = x.permute(0, 2, 1)  # Change to [batch, features, seq_len] for conv1d

#         proj_out = self.projection(x)
#         res_out = self.residual(x)
        
#         return proj_out + res_out





class ProjectionAttentionWithFFT(nn.Module):
    def __init__(self, d_model, n_heads, c_out, dropout_rate=0.1):
        super(ProjectionAttentionWithFFT, self).__init__()

        # Enhanced convolutional layers without reducing dimensions too quickly
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            # Use convolution layers to adjust features without reducing dimension drastically
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        

        # Linear projection layers
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.LayerNorm(d_model//2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(d_model//2, d_model//4),
            nn.GELU(),
            nn.LayerNorm(d_model//4),
            nn.Dropout(dropout_rate),

            nn.Linear(d_model//4, c_out)
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

        # Adding a residual connection to help maintain original information flow
        self.residual = nn.Linear(d_model, c_out) if d_model != c_out else nn.Identity()

    def forward(self, x):
        y=x
        # Permute dimensions to align with Conv1d input requirements
        x = x.permute(0, 2, 1)  # Change to [batch, features, seq_len]

        # Apply convolutional layers
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)  # Revert to [batch, seq_len, features]

       

        # Apply projection and combine with residual
        proj_out = self.projection(x)

        res_out = self.residual(y)
        
        gate = self.gate(x)
        out = gate * proj_out + (1 - gate) * res_out
        
        return out

























# *****
# thsi is the best model so far we have



# class ProjectionAttentionWithFFT(nn.Module):
#     def __init__(self, d_model, n_heads, c_out, dropout_rate=0.1):
#         super(ProjectionAttentionWithFFT, self).__init__()
#         self.multihead_attn = nn.MultiheadAttention(d_model, n_heads)
        
        
#         self.projection = nn.Sequential(
#             nn.Linear(d_model, d_model//2),
#             nn.GELU(), 
#             my_Layernorm(d_model//2),
#             nn.Dropout(dropout_rate),
            

            
#             nn.Linear(d_model//2, d_model//4),
#             nn.GELU(),  
#             my_Layernorm(d_model//4),
#             nn.Dropout(dropout_rate),


#             nn.Linear(d_model//4, c_out)  
#         )

#     def forward(self, x):

#         x = self.projection(x)  # Simplified projection pathway
        
#         return x





