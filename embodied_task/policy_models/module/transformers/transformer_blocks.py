from matplotlib.pyplot import cla
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from inspect import isfunction

from typing import Optional, Tuple

import logging
import math 
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
import einops

from .position_embeddings import *


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)



# RMSNorm -- Better, simpler alternative to LayerNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale, self.eps = dim**-0.5, eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


# SwishGLU -- A Gated Linear Unit (GLU) with the Swish activation; always better than GELU MLP!
class SwishGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.act, self.project = nn.SiLU(), nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected, gate = self.project(x).tensor_split(2, dim=-1)
        return projected * self.act(gate)



class Attention(nn.Module):

    def __init__(
            self, 
            n_embd: int,
            n_head: int,
            attn_pdrop: float,
            resid_pdrop: float,
            block_size: int,
            causal: bool = False,
            bias=False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
            rotary_emb_dim = None,
            rotary_xpos_scale_base = 512,
            rotary_interpolation_factor = 1.,
        ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd
        self.causal = causal
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))
        self.use_rot_embed = use_rot_embed
        if self.use_rot_embed:
        # Update (12/2022): Rotary embedding has since been hugely successful, widely adopted in many large language models, including the largest in the world, PaLM. 
        # However, it has been uncovered in the ALiBi paper that rotary embeddings cannot length extrapolate well. 
        # This was recently addressed in <a href="https://arxiv.org/abs/2212.10554v1">a Microsoft research paper</a>. 
        # They propose a way to unobtrusively add the same decay as in ALiBi, and found that this resolves the extrapolation problem.
        # You can use it in this repository by setting `rotary_xpos = True`. Like ALiBi, it would enforce the attention to be local. You can set the receptive field with `rotary_xpos_scale_base` value, which defaults to `512`
            rotary_emb_dim = max(default(rotary_emb_dim, self.n_head // 2), 32)
            self.rotary_pos_emb = RotaryEmbedding(
                rotary_emb_dim, 
                use_xpos = rotary_xpos, 
                xpos_scale_base = rotary_xpos_scale_base, 
                interpolate_factor = rotary_interpolation_factor, 
            ) 

    def forward(self, x, context=None, custom_attn_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # if the context is not None we do cross-attention othberwise self=attention
        # cross attention computes the query from x and the keys and values are from the context
        if context is not None:
            k = self.key(context).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = self.value(context).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        else:
            k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # apply rotary stuff here if needed:
        if self.use_rot_embed:
            q = self.rotary_pos_emb.rotate_queries_or_keys(q)
            k = self.rotary_pos_emb.rotate_queries_or_keys(k)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=custom_attn_mask, dropout_p=self.attn_dropout.p if self.training else 0, is_causal=self.causal)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.causal:
                if custom_attn_mask is not None:
                    att = att.masked_fill(custom_attn_mask == 0, float('-inf'))
                else:
                    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    

class MLP(nn.Module):

    def __init__(
            self, 
            n_embd: int,
            bias: bool,
            dropout: float = 0
        ):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(
            self, 
            n_embd: int, 
            n_heads: int, 
            attn_pdrop: float, 
            resid_pdrop: float, 
            mlp_pdrop: float,
            block_size: int, 
            causal: bool,
            use_cross_attention: bool = False,
            use_rot_embed: bool=False,
            rotary_xpos: bool = False,
            bias: bool = False, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        ):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = Attention(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size, causal, bias, use_rot_embed, rotary_xpos)
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_att = Attention(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size, causal, bias, use_rot_embed, rotary_xpos)
            self.ln3 = nn.LayerNorm(n_embd)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias, mlp_pdrop)

    def forward(self, x, context=None, custom_attn_mask=None):
        x = x + self.attn(self.ln_1(x), custom_attn_mask=custom_attn_mask)
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.ln3(x), context, custom_attn_mask=custom_attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x



class CrossAttentionOnlyBlock(nn.Module):

    def __init__(
            self, 
            n_embd: int, 
            n_heads: int, 
            attn_pdrop: float, 
            resid_pdrop: float, 
            mlp_pdrop: float,
            block_size: int, 
            causal: bool,
            use_rot_embed: bool=False,
            rotary_xpos: bool = False,
            bias: bool = False, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        ):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.cross_att = Attention(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size, causal, bias, use_rot_embed, rotary_xpos)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias, mlp_pdrop)

    def forward(self, x, context=None, custom_attn_mask=None):
        x = x + self.cross_att(self.ln_1(x), context, custom_attn_mask=custom_attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class AdaLNZero(nn.Module):
    """
    AdaLN-Zero modulation for conditioning.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # Initialize weights and biases to zero
        # nn.init.zeros_(self.modulation[1].weight)
        # nn.init.zeros_(self.modulation[1].bias)

    def forward(self, c):
        return self.modulation(c).chunk(6, dim=-1)

def modulate(x, shift, scale):
    return shift + (x * (scale))


class ConditionedBlock(Block):
    """
    Block with AdaLN-Zero conditioning.
    """
    def __init__(
            self, 
            n_embd, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop, 
            block_size, 
            causal, 
            film_cond_dim,
            use_cross_attention=False, 
            use_rot_embed=False, 
            rotary_xpos=False, 
            bias=False # and any other arguments from the Block class
        ):
        super().__init__(n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop, block_size, causal,
                         use_cross_attention=use_cross_attention, 
                         use_rot_embed=use_rot_embed, 
                         rotary_xpos=rotary_xpos, 
                         bias=bias)
        self.adaLN_zero = AdaLNZero(film_cond_dim)

    def forward(self, x, c, context=None, custom_attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_zero(c)
        
        # Attention with modulation
        x_attn = self.ln_1(x)
        x_attn = modulate(x_attn, shift_msa, scale_msa)
        x = x + gate_msa * self.attn(x_attn, custom_attn_mask=custom_attn_mask)
        
        # Cross attention if used
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.ln3(x), context, custom_attn_mask=custom_attn_mask)
        
        # MLP with modulation
        x_mlp = self.ln_2(x)
        x_mlp = modulate(x_mlp, shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_mlp)
        
        return x

class NoiseBlock(Block):
    """
    Block with AdaLN-Zero conditioning.
    """
    def __init__(
            self, 
            n_embd, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop, 
            block_size, 
            causal, 
            use_cross_attention=False, 
            use_rot_embed=False, 
            rotary_xpos=False, 
            bias=False # and any other arguments from the Block class
        ):
        super().__init__(n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop, block_size, causal,
                         use_cross_attention=use_cross_attention, 
                         use_rot_embed=use_rot_embed, 
                         rotary_xpos=rotary_xpos, 
                         bias=bias)

    def forward(self, x, c, context=None, custom_attn_mask=None):
        
        x = x + self.attn(self.ln_1(x) + c, custom_attn_mask=custom_attn_mask)
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.ln3(x) + c, context, custom_attn_mask=custom_attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
    

class TransformerEncoder(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            n_heads: int, 
            attn_pdrop: float,  
            resid_pdrop: float, 
            n_layers: int, 
            block_size: int,
            bias: bool = False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
            mlp_pdrop: float = 0,
        ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[Block(
            embed_dim, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop,
            block_size,
            causal=False, 
            use_rot_embed=use_rot_embed,
            rotary_xpos=rotary_xpos,
            bias=bias
            ) 
            for _ in range(n_layers)]
        )
        self.ln = LayerNorm(embed_dim, bias)

    def forward(self, x, custom_attn_mask=None):
        for layer in self.blocks:
            x = layer(x, custom_attn_mask=custom_attn_mask)
        x = self.ln(x)
        return x
    
    
class TransformerEncoderInterleaved(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            n_heads: int, 
            attn_pdrop: float,  
            resid_pdrop: float, 
            n_layers: int, 
            block_size: int,
            bias: bool = False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
            mlp_pdrop: float = 0,
        ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[Block(
            embed_dim, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop,
            block_size,
            causal=False, 
            use_rot_embed=use_rot_embed,
            rotary_xpos=rotary_xpos,
            bias=bias
            ) 
            for _ in range(n_layers)]
        )
        self.ln = LayerNorm(embed_dim, bias)

    def forward(self, x):
        outputs = []
        for layer in self.blocks:
            x = layer(x)
            outputs.append(x)
        x = self.ln(x)
        outputs.pop(-1)
        outputs.append(x)
        return outputs
    

class TransformerFiLMEncoder(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            n_heads: int, 
            attn_pdrop: float,  
            resid_pdrop: float, 
            n_layers: int, 
            block_size: int,
            film_cond_dim: int,
            bias: bool = False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
            mlp_pdrop: float = 0,
        ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ConditionedBlock(
            embed_dim, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop,
            block_size,
            causal=False, 
            use_rot_embed=use_rot_embed,
            rotary_xpos=rotary_xpos,
            bias=bias,
            film_cond_dim=film_cond_dim
            ) 
            for _ in range(n_layers)]
        )
        self.ln = LayerNorm(embed_dim, bias)

    def forward(self, x, c):
        for layer in self.blocks:
            x = layer(x, c)
        x = self.ln(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            n_heads: int, 
            attn_pdrop: float,  
            resid_pdrop: float, 
            n_layers: int, 
            block_size: int,
            bias: bool = False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
            mlp_pdrop: float = 0,
            use_cross_attention: bool = True,
        ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[Block(
            embed_dim, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop,
            block_size,
            causal=True, 
            use_cross_attention=use_cross_attention,
            use_rot_embed=use_rot_embed,
            rotary_xpos=rotary_xpos,
            bias=bias
            ) 
            for _ in range(n_layers)]
        )
        self.ln = LayerNorm(embed_dim, bias)

    def forward(self, x, cond=None, custom_attn_mask=None):
        for layer in self.blocks:
            x = layer(x, cond, custom_attn_mask=custom_attn_mask)
        x = self.ln(x)
        return x



class TransformerFiLMDecoder(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            n_heads: int, 
            attn_pdrop: float,  
            resid_pdrop: float, 
            n_layers: int, 
            block_size: int,
            film_cond_dim: int,
            bias: bool = False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
            mlp_pdrop: float = 0,
            use_cross_attention: bool = True,
            use_noise_encoder: bool = False,
            kwargs: Optional[DictConfig] = None,
        ):
        super().__init__()
        if use_noise_encoder:
            self.blocks = nn.Sequential(
                *[NoiseBlock(
                embed_dim, 
                n_heads, 
                attn_pdrop, 
                resid_pdrop, 
                mlp_pdrop,
                block_size,
                causal=True, 
                use_cross_attention=use_cross_attention,
                use_rot_embed=use_rot_embed,
                rotary_xpos=rotary_xpos,
                bias=bias,
                ) 
                for _ in range(n_layers)]
            )
        else:
            self.blocks = nn.Sequential(
                *[ConditionedBlock(
                embed_dim, 
                n_heads, 
                attn_pdrop, 
                resid_pdrop, 
                mlp_pdrop,
                block_size,
                causal=True, 
                use_cross_attention=use_cross_attention,
                use_rot_embed=use_rot_embed,
                rotary_xpos=rotary_xpos,
                bias=bias,
                film_cond_dim=film_cond_dim,
                ) 
                for _ in range(n_layers)]
            )
        self.ln = LayerNorm(embed_dim, bias)

    def forward(self, x, c, cond=None, custom_attn_mask=None):
        for layer in self.blocks:
            x = layer(x, c, cond, custom_attn_mask=custom_attn_mask)
        x = self.ln(x)
        return x


class TransformerFiLMDecoderInterleaved(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            n_heads: int, 
            attn_pdrop: float,  
            resid_pdrop: float, 
            n_layers: int, 
            block_size: int,
            film_cond_dim: int,
            bias: bool = False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
            mlp_pdrop: float = 0,
            use_cross_attention: bool = True,
            use_noise_encoder: bool = False,
            kwargs: Optional[DictConfig] = None,
        ):
        super().__init__()
        if use_noise_encoder:
            self.blocks = nn.Sequential(
                *[NoiseBlock(
                embed_dim, 
                n_heads, 
                attn_pdrop, 
                resid_pdrop, 
                mlp_pdrop,
                block_size,
                causal=True, 
                use_cross_attention=use_cross_attention,
                use_rot_embed=use_rot_embed,
                rotary_xpos=rotary_xpos,
                bias=bias,
                ) 
                for _ in range(n_layers)]
            )
        else:
            self.blocks = nn.Sequential(
                *[ConditionedBlock(
                embed_dim, 
                n_heads, 
                attn_pdrop, 
                resid_pdrop, 
                mlp_pdrop,
                block_size,
                causal=True, 
                use_cross_attention=use_cross_attention,
                use_rot_embed=use_rot_embed,
                rotary_xpos=rotary_xpos,
                bias=bias,
                film_cond_dim=film_cond_dim,
                ) 
                for _ in range(n_layers)]
            )
        self.ln = LayerNorm(embed_dim, bias)

    def forward(self, x, c, cond=None, custom_attn_mask=None):
        for idx, layer in enumerate(self.blocks):
            cond_tokens =cond[idx]
            x = layer(x, c, cond_tokens, custom_attn_mask=custom_attn_mask)
        x = self.ln(x)
        return x


class TransformerCrossAttentionEncoder(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            n_heads: int, 
            attn_pdrop: float,  
            resid_pdrop: float, 
            n_layers: int, 
            block_size: int,
            bias: bool = False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
            mlp_pdrop: float = 0,
            use_cross_attention: bool = True,
        ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[Block(
            embed_dim, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop,
            block_size,
            causal=False, 
            use_cross_attention=use_cross_attention,
            use_rot_embed=use_rot_embed,
            rotary_xpos=rotary_xpos,
            bias=bias
            ) 
            for _ in range(n_layers)]
        )
        self.ln = LayerNorm(embed_dim, bias)

    def forward(self, x, cond=None, custom_attn_mask=None):
        for layer in self.blocks:
            x = layer(x, cond, custom_attn_mask=custom_attn_mask)
        x = self.ln(x)
        return x
    

class TransformerCrossAttentionOnlyEncoder(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            n_heads: int, 
            attn_pdrop: float,  
            resid_pdrop: float, 
            n_layers: int, 
            block_size: int,
            bias: bool = False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
            mlp_pdrop: float = 0,
            use_cross_attention: bool = True,
        ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[CrossAttentionOnlyBlock(
            embed_dim, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop,
            block_size,
            causal=False, 
            use_rot_embed=use_rot_embed,
            rotary_xpos=rotary_xpos,
            bias=bias
            ) 
            for _ in range(n_layers)]
        )
        self.ln = LayerNorm(embed_dim, bias)

    def forward(self, x, cond=None, custom_attn_mask=None):
        for layer in self.blocks:
            x = layer(x, cond, custom_attn_mask=custom_attn_mask)
        x = self.ln(x)
        return x

# As defined in Set Transformers () -- basically the above, additionally taking in
# a set of $k$ learned "seed vectors" that are used to "pool" information.
class MAPAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int) -> None:
        """Multi-Input Multi-Headed Attention Operation"""
        super().__init__()
        assert embed_dim % n_heads == 0, "`embed_dim` must be divisible by `n_heads`!"
        self.n_heads, self.scale = n_heads, (embed_dim // n_heads) ** -0.5

        # Projections (no bias) --> separate for Q (seed vector), and KV ("pool" inputs)
        self.q, self.kv = nn.Linear(embed_dim, embed_dim, bias=False), nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, seed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        (B_s, K, C_s), (B_x, N, C_x) = seed.shape, x.shape
        assert C_s == C_x, "Seed vectors and pool inputs must have the same embedding dimensionality!"

        # Project Seed Vectors to `queries`
        q = self.q(seed).reshape(B_s, K, self.n_heads, C_s // self.n_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B_x, N, 2, self.n_heads, C_x // self.n_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # Attention --> compute weighted sum over values!
        scores = q @ (k.transpose(-2, -1) * self.scale)
        attn = scores.softmax(dim=-1)
        vals = (attn @ v).transpose(1, 2).reshape(B_s, K, C_s)

        # Project back to `embed_dim`
        return self.proj(vals)


class MAPBlock(nn.Module):
    def __init__(
        self,
        n_latents: int,
        embed_dim: int,
        n_heads: int,
        output_dim: None,
        mlp_ratio: float = 4.0,
        do_rms_norm: bool = True,
        do_swish_glu: bool = True,
    ) -> None:
        """Multiheaded Attention Pooling Block -- note that for MAP, we adopt earlier post-norm conventions."""
        super().__init__()
        self.n_latents, self.embed_dim, self.n_heads = n_latents, embed_dim, 2 * n_heads

        self.embed_dim = output_dim
        # Projection Operator
        self.projection = nn.Linear(embed_dim, self.embed_dim)

        # Initialize Latents
        self.latents = nn.Parameter(torch.zeros(self.n_latents, self.embed_dim))
        nn.init.normal_(self.latents, std=0.02)

        # Custom MAP Attention (seed, encoder outputs) -> seed
        self.attn_norm = RMSNorm(self.embed_dim) if do_rms_norm else nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.attn = MAPAttention(self.embed_dim, n_heads=self.n_heads)
        if output_dim is None:
            output_dim = self.embed_dim
        # Position-wise Feed-Forward Components
        self.mlp_norm = RMSNorm(self.embed_dim) if do_rms_norm else nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            # Handle SwishGLU vs. GELU MLP...
            (
                SwishGLU(self.embed_dim, int(mlp_ratio * self.embed_dim))
                if do_swish_glu
                else nn.Sequential(nn.Linear(self.embed_dim, int(mlp_ratio * self.embed_dim)), nn.GELU())
            ),
            nn.Linear(int(mlp_ratio * self.embed_dim), self.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latents = repeat(self.latents, "n_latents d -> bsz n_latents d", bsz=x.shape[0])
        latents = self.attn_norm(latents + self.attn(latents, self.projection(x)))
        latents = self.mlp_norm(latents + self.mlp(latents))
        return latents.squeeze(dim=1)


class SiamneseDecoder(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            n_heads: int, 
            attn_pdrop: float,  
            resid_pdrop: float, 
            n_layers: int, 
            block_size: int,
            bias: bool = False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
            mlp_pdrop: float = 0,
            use_cross_attention: bool = True,
        ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[Block(
            embed_dim, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop,
            block_size,
            causal=False, 
            use_cross_attention=use_cross_attention,
            use_rot_embed=use_rot_embed,
            rotary_xpos=rotary_xpos,
            bias=bias
            ) 
            for _ in range(n_layers)]
        )
        self.ln = LayerNorm(embed_dim, bias)

    def forward(self, x, cond=None, custom_attn_mask=None):
        for layer in self.blocks:
            x = layer(x, cond, custom_attn_mask=custom_attn_mask)
        x = self.ln(x)
        return x
    

class ClipStyleProjection(nn.Module):
    
    def __init__(self, clip_style, token_dim=384, clip_token_index=0, num_token=4):
        super(ClipStyleProjection, self).__init__()
        self.clip_style = clip_style
        self.clip_token_index = clip_token_index
        if clip_style == 'map' or clip_style == 'map_state_only':
            self.latent_proj = MAPBlock(1, token_dim, 8, output_dim=token_dim)
        elif clip_style == 'mean_pooling' or clip_style == 'mean_pool_state_only':
            self.latent_proj = MeanPooling(token_dim)
        elif clip_style == 'mlp':
            self.latent_proj = nn.Sequential(
                nn.Linear(num_token * token_dim, token_dim),
                nn.LayerNorm(token_dim), 
                nn.Tanh()
            )
        elif clip_style == 'single_token':
            self.latent_proj = nn.Identity() 
        elif clip_style == 'multihead':
            self.latent_proj = nn.Identity()  # No projection needed
        else:
            raise ValueError("Invalid clip_style. Expected 'map', 'mean_pooling', or 'single_token' or 'multihead'.")
        # print(self.clip_style)

    def forward(self, x):
        # print('clip style is ' + self.clip_style)
        # print(f'x shape {x.shape}')
        if self.clip_style == 'single_token':
            x = x[:, self.clip_token_index, :]
        elif self.clip_style == 'map_state_only' or self.clip_style == 'mean_pool_state_only':
            x = x[:, 1:]
        elif self.clip_style == 'mlp':
            # print('reshaping before clip')
            x = einops.rearrange(x, 'b t d -> b (t d)')
        # print(x.shape)
        return self.latent_proj(x)


class MeanPooling(nn.Module):
    def __init__(self, token_dim):
        super(MeanPooling, self).__init__()
        self.token_dim = token_dim

    def forward(self, x):
        return x.mean(dim=1).view(-1, self.token_dim)

