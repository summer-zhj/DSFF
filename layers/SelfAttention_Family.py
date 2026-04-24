import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat
import torch.nn.functional as F
import math


class HybridAttention(nn.Module):
    def __init__(self, d_model, n_heads, kernel_size, attention_dropout=0.1, scale=None, output_attention=False):
        super(HybridAttention, self).__init__()

        # 全局注意力部分
        self.global_attention = FullAttention(mask_flag=False, scale=scale, attention_dropout=attention_dropout,
                                              output_attention=output_attention)

        # 局部注意力部分
        self.local_attention = LocalAttention(mask_flag=True, window_size=5, scale=scale, attention_dropout=attention_dropout,
                                              output_attention=output_attention)

        self.dropout = nn.Dropout(attention_dropout)

        self.output_attention = output_attention

    def forward(self, queries, keys, values, attn_mask):
        # 全局注意力分支
        global_attention_output, global_attention_weights = self.global_attention(queries, keys, values, attn_mask)

        # 局部注意力分支
        local_attention_output,local_attention_weights = self.local_attention(queries, keys, values, attn_mask)

        # 你可以根据需要选择不同的融合方式，这里我们使用加法融合
        #print(global_attention_output.shape, local_attention_output.shape)

        combined_output = global_attention_output + local_attention_output

        # 如果需要返回注意力权重
        if self.output_attention:
            return combined_output, (global_attention_weights, local_attention_weights)
        else:
            return combined_output, None
class LocalAttention(nn.Module):
    def __init__(self, window_size=5, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(LocalAttention, self).__init__()
        self.window_size = window_size  # 设置窗口大小
        self.mask_flag = mask_flag
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape  # B: batch size, L: sequence length, H: heads, E: embedding dimension
        _, S, _, D = values.shape  # S: sequence length of values (may differ from keys length)

        scale = self.scale or 1. / sqrt(E)

        # Create local attention mask for the sliding window
        local_mask = self.create_local_mask(B, L, self.window_size, queries.device)

        # Ensure that the local_mask is the same shape as the attention scores
        local_mask = local_mask.unsqueeze(1).expand(-1, H, -1, -1)

        # Compute attention scores (query * key)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)


        if self.mask_flag:
            # Apply the local mask to scores, where each query only attends to a local window of keys
            scores.masked_fill_(local_mask, -np.inf)


        # Apply softmax and dropout to get attention weights
        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        # Compute the weighted sum of values (V) using the attention weights
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

    def create_local_mask(self, B, L, window_size, device):
        """
        Create a local attention mask for each query in a sliding window fashion.
        Each query will only attend to keys within a window of size 'window_size'.
        """
        mask = torch.zeros((B, L, L), device=device)
        for i in range(L):
            start = max(i - window_size // 2, 0)
            end = min(i + window_size // 2, L)
            mask[:, i, start:end] = 1
        return mask == 0  # Invert mask (0 means we mask this part out)

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
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
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

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
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
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
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

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
            attn_mask,
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None

class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out

class TriangularCausalMask:
    def __init__(self, batch_size, seq_len, device):
        # 创建上三角的因果掩码，1表示不可见
        self.mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        # 扩展为 [B, 1, L, L]，适用于批次大小 B
        self.mask = self.mask.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)

    def apply(self, scores):
        # 扩展掩码到 [B, H, L, L]，与 scores 的形状对齐
        # 假设 scores 的形状是 [B, H, L, S]，即 [批次大小, 注意力头数, 查询长度, 键长度]
        B, H, L, S = scores.shape
        # 扩展掩码到 [B, H, L, L]
        mask_expanded = self.mask.expand(B, H, L, L)  # 扩展掩码
        # 应用掩码，确保未来时间步对当前时间步不可见
        scores.masked_fill_(mask_expanded, -np.inf)
        return scores
class CausalAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(CausalAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # 计算查询（queries）和键（keys）之间的得分
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # [B, H, L, S]

        if self.mask_flag:
            if attn_mask is None:
                # 默认使用因果掩码
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            # 对得分矩阵应用因果掩码，确保每个位置只能看到自己和之前的时间步
            scores = attn_mask.apply(scores)  # [B, H, L, S]

        # 计算注意力权重，并应用 Dropout
        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # [B, H, L, S]

        # 计算加权的值
        V = torch.einsum("bhls,bshd->blhd", A, values)  # [B, L, H, D]

        if self.output_attention:
            return V.contiguous(), A  # 返回加权的值和注意力权重
        else:
            return V.contiguous(), None  # 仅返回加权的值
