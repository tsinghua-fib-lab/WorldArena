# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import warnings
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from wan.modules.block_attention import BLHD_flex_with_mask
from wan.modules.attention import flash_attention
try:
    from flash_attn.cute.interface import flash_attn_func
    CUTE_FA = True
except:
    CUTE_FA = False
    warnings.warn('using FA2')
    from flash_attn.flash_attn_interface import flash_attn_func
__all__ = ['WanModelCausal']

def fa_switch(q, k, v):
    if CUTE_FA:
        o, lse = flash_attn_func(q, k, v)
        return o
    else:
        return flash_attn_func(q, k, v)

def ckpt_wrapper(module):
    def ckpt_forward(*inputs, **kwargs):
        outputs = module(*inputs, **kwargs)
        return outputs

    return ckpt_forward

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()

@torch.amp.autocast('cuda', enabled=False)
def rope_apply_one(x, grid_size, freqs_in, block_idx=None, block_size=None):
    B, L, H, D = x.shape
    n, c = x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs_in.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid_size.tolist()
    if block_idx is not None:
        f = max(f, block_idx + 1)
    seq_len = f * h * w
    # precompute multipliers
    # 先调整维度顺序：[B, L, H, D] -> [L, B, H, D] -> [seq_len, B*H, D//2, 2]
    x_reordered = x.permute(1, 0, 2, 3)  # [L, B, H, D]
    x_i = torch.view_as_complex(x_reordered.to(torch.float64).reshape(L, n * B, -1, 2))
    freqs_i = torch.cat([
        freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(seq_len, 1, -1)
    
    if block_idx is not None:
        assert x_i.shape[0] == block_size, "when do perframe rope, block_size must aligh with seq len"
        freqs_i = freqs_i[block_idx * block_size:(block_idx + 1) * block_size, :, :]
    # apply rotary embedding
    x_i = torch.view_as_real(x_i * freqs_i).flatten(2).reshape(L, B, H, D)
    # 调整回原来的维度顺序：[L, B, H, D] -> [B, L, H, D]
    x_i = x_i.permute(1, 0, 2, 3)
    return x_i.float()


@torch.amp.autocast('cuda', enabled=False)
def rope_apply_chunk(x, grid_size, freqs_in, kv_num_block=None, block_size=None):
    # import IPython; IPython.embed()
    x_num_block = x.shape[1] // block_size
    B, L, H, D = x.shape
    n, c = x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs_in.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid_size.tolist()
    if kv_num_block is not None:
        f = max(f, kv_num_block + 1 + x_num_block)
    seq_len = f * h * w
    # precompute multipliers
    # 先调整维度顺序：[B, L, H, D] -> [L, B, H, D] -> [seq_len, B*H, D//2, 2]
    x_reordered = x.permute(1, 0, 2, 3)  # [L, B, H, D]
    x_i = torch.view_as_complex(x_reordered.to(torch.float64).reshape(L, n * B, -1, 2))
    freqs_i = torch.cat([
        freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(seq_len, 1, -1)
    
    if kv_num_block is not None:
        freqs_i = freqs_i[kv_num_block * block_size:(kv_num_block + x_num_block) * block_size, :, :]
    # apply rotary embedding
    x_i = torch.view_as_real(x_i * freqs_i).flatten(2).reshape(L, B, H, D)
    # 调整回原来的维度顺序：[L, B, H, D] -> [B, L, H, D]
    x_i = x_i.permute(1, 0, 2, 3)
    return x_i.float()


def test_rope_apply():
    # 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    f, h, w = 4, 4, 4
    B, L, H, D = 1, f * h * w, 8, 64
    d = D
    freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)
    freqs = freqs.to(device)
    x = torch.randn(B, L, H, D, device=device, dtype=torch.float32)
    x = torch.cat([x, x], dim=0)
    grid_size = torch.tensor([f, h, w])
    x_out = rope_apply_one(x, grid_size, freqs)
    
    # test1, front half equals to the back half
    assert torch.allclose(x_out[0, :, :, :], x_out[1, :, :, :], atol=1e-5), "Front half does not equal to back half"
    
    # test2, align with rope_apply
    x_out2 = rope_apply(x, grid_size.unsqueeze(0), freqs)
    assert torch.allclose(x_out, x_out2, atol=1e-5), "Output does not match rope_apply output"
    
    # test3, with frame index parameter
    block_size = h * w
    block_idx = 0
    for block_idx in range(f):
        # test with block_idx
        x_0 = x[:, block_size * block_idx:block_size * (block_idx + 1), :, :]
        x_0_out = rope_apply_one(x_0, grid_size, freqs, block_idx=block_idx, block_size=block_size)
        assert torch.allclose(x_0_out, x_out[:, block_size * block_idx:block_size * (block_idx + 1), :, :], atol=1e-5), "Output does not match with block_idx"
    print("test 3 passed diff:", (x_0_out - x_out[:, block_size * block_idx:block_size * (block_idx + 1), :, :]).norm())


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.cached_k = None
        self.cached_v = None

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    
    def clean_cache(self):
        self.cached_k = None
        self.cached_v = None
        torch.cuda.empty_cache()
    
    def qkv_fn(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v
    
    def prefill(self, q, k, v, attention_mask = None):
        assert attention_mask is not None, 'Attention mask must be provided for prefill'
        self.cached_k = k
        self.cached_v = v
        o = BLHD_flex_with_mask(q, k, v, attention_mask)
        return o

    def pop_kvcache(self, num_to_pop):
        # pop from back
        assert self.cached_k is not None
        self.cached_k = self.cached_k[:, :-num_to_pop, :, :]
        self.cached_v = self.cached_v[:, :-num_to_pop, :, :]
    
    
    def chunk_prefill(self, q, k, v, attention_mask, grid_sizes, freqs, block_size):
        cur_kv_len = self.cached_k.shape[1]
        cur_kv_num_block = cur_kv_len // block_size
        q = rope_apply_chunk(q, grid_sizes[0], freqs, kv_num_block=cur_kv_num_block, block_size=block_size).to(v.dtype)
        k = rope_apply_chunk(k, grid_sizes[0], freqs, kv_num_block=cur_kv_num_block, block_size=block_size).to(v.dtype)
        self.cached_k = torch.cat([self.cached_k, k], dim=1)
        self.cached_v = torch.cat([self.cached_v, v], dim=1)
        x = BLHD_flex_with_mask(q, self.cached_k, self.cached_v, attention_mask)
        x = x.flatten(2)
        x = self.o(x)
        return x
    
    def forward(self, x, grid_sizes, freqs,
        attention_mask=None,
        block_size=None,
        cache=False,
        prefill=False,
        chunk_prefill=False,
        block_idx=None,
        ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        q, k, v = self.qkv_fn(x)
        if chunk_prefill:
            return self.chunk_prefill(q, k, v, attention_mask, grid_sizes, freqs, block_size)

        q = rope_apply_one(q, grid_sizes[0], freqs, block_idx=block_idx, block_size=block_size).to(v.dtype)
        k = rope_apply_one(k, grid_sizes[0], freqs, block_idx=block_idx, block_size=block_size).to(v.dtype)
        
        if prefill:
            assert attention_mask is not None
            x = self.prefill(q, k, v, attention_mask)
        elif cache: # cache forward
            assert k.shape[1] == block_size
            assert k.shape[0] == 1, 'current only support shared kv across batch(i.e. cfg)'
            if self.cached_k is None:
                self.cached_k = k[:, :block_size, :, :]
                self.cached_v = v[:, :block_size, :, :]
            else:
                self.cached_k = torch.cat([self.cached_k, k[:, :block_size, :, :]], dim=1)
                self.cached_v = torch.cat([self.cached_v, v[:, :block_size, :, :]], dim=1)
            x = fa_switch(q, self.cached_k, self.cached_v)
        else:
            if self.cached_k is not None:
                assert block_size is not None and block_idx is not None
                B = x.shape[0]
                cache_k = self.cached_k.repeat(B, 1, 1, 1)
                cache_v = self.cached_v.repeat(B, 1, 1, 1)
                k = torch.cat((cache_k, k[:, -block_size:, :, :]), dim=1)
                v = torch.cat((cache_v, v[:, -block_size:, :, :]), dim=1)
            if attention_mask is not None:
                x = BLHD_flex_with_mask(q, k, v, attention_mask)
            else:
                x = fa_switch(q, k, v)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        grid_sizes,
        freqs,
        context,
        context_lens,
        attention_mask=None,
        block_size=None,
        cache=False,
        prefill=False,
        chunk_prefill=False,
        block_idx=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            None(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            grid_sizes, freqs, 
            attention_mask=attention_mask,
            block_size=block_size, cache=cache, prefill=prefill, chunk_prefill=chunk_prefill,
            block_idx=block_idx)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class WanModelCausal(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.gradient_checkpoint = False

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

    
    def clean_cache(self):
        for block in self.blocks:
            block.self_attn.clean_cache()
    
    def kvcache_len(self):
        k_cache = self.blocks[0].self_attn.cached_k
        if k_cache is None:
            return 0
        else:
            return k_cache.shape[1]
    
    def pop_kvcache(self, num_to_pop):
        assert self.kvcache_len() >= num_to_pop
        for block in self.blocks:
            block.self_attn.pop_kvcache(num_to_pop)

    def fully_shard(self, mesh: DeviceMesh):
        for i, block in enumerate(self.blocks):
            reshard_after_forward = i < len(self.blocks) - 1
            fully_shard(block, mesh=mesh, reshard_after_forward=reshard_after_forward)
        fully_shard(self.head, mesh=mesh, reshard_after_forward=True)
        fully_shard(self.time_embedding, mesh=mesh, reshard_after_forward=True)

    def forward(
        self,
        x,
        t,
        context,
        attention_mask=None,
        block_size=None,
        cache=False,
        prefill=False,
        chunk_prefill=False,
        block_idx=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (Tensor):
                Batch of video tensors, each with shape [B, C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B, seq_len], dtype=torch.float32
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]

        Returns:
            Tensor:
                List of denoised video tensors with original input shapes [B, C_out, F, H / 8, W / 8]
        """
        assert not (prefill and chunk_prefill), "prefill and chunk_prefill cannot be True at the same time"
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        # embeddings
        x = self.patch_embedding(x)
        
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long, device=device)
        grid_sizes = grid_size.unsqueeze(0)
        x = x.flatten(2).transpose(1, 2)  # B, L, D
        B, seq_len, D = x.shape

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # arguments
        kwargs = dict(
            e=e0,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            attention_mask=attention_mask,
            block_size=block_size,
            cache=cache,
            prefill=prefill,
            chunk_prefill=chunk_prefill,
            block_idx=block_idx,
            )
        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify_batch(x, grid_size)
        return x.float()


    def unpatchify_batch(self, x, grid_size):
        c = self.out_dim
        B = x.size(0)
        x = x.view(B, *grid_size, *self.patch_size, c)
        x = torch.einsum('bfhwpqrc->bcfphqwr', x)
        x = x.reshape(B, c, *[i * j for i, j in zip(grid_size, self.patch_size)])
        return x

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)


if __name__ == '__main__':
    # Example usage
    model = WanModelCausal()
    x = [torch.randn(1, 16, 8, 64, 64)]  # Example input tensor
    t = torch.tensor([0.5])  # Example timestep
    context = [torch.randn(10, 4096)]  # Example text embedding
    output = model(x, t, context, seq_len=8)
    print(output[0].shape)  # Should print the shape of the output tensor