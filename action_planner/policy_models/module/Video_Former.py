# This code is referenced from https://github.com/dhansmair/flamingo-mini

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops_exts import rearrange_many
from policy_models.module.transformers.utils import feed_forward_layer
from torch import einsum, nn


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        use_cross_attn=False,
        y_dim=512,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        attn_mask=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_mask = attn_mask
        self.use_cross_attn = use_cross_attn
        if self.use_cross_attn:
            # print('use_cross_attn')
            self.y_kv = nn.Linear(y_dim, dim * 2, bias=qkv_bias)
            self.y_k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
            self.gate = nn.Parameter(torch.zeros([self.num_heads]))

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            if self.attn_mask is not None:
                self.attn_mask = self.attn_mask.to(x.device)
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=self.attn_mask,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        if self.use_cross_attn:
            # print('y_shape:',y.shape)
            N_y = y.shape[1]
            y_kv = (
                self.y_kv(y)
                .reshape(B, N_y, 2, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            y_k, y_v = y_kv.unbind(0)
            y_k = self.y_k_norm(y_k)
            y_out = F.scaled_dot_product_attention(
                q,
                y_k,
                y_v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            # print('y_out_shape:', y_out.shape)
            y_out = y_out * self.gate.tanh().view(1, -1, 1, 1)
            x = x + y_out

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PerceiverAttentionLayer(nn.Module):
    """Perceiver Attention Layer"""

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        # trainable components of PerceiverAttentionLayer
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, features, latents):
        """Latent vectors are cross-attending to the visual features x

        Args:
            features: Batch of visual features with shape (batch_size, n_features, dim)
            latents: Latent learnt vectors which are used to compute queries with shape (batch_size, n_latents, dim)

        Returns:
            Attention score with shape (batch_size, n_latents, dim)
        """
        assert features.ndim == 3
        assert latents.ndim == 3
        assert features.shape[0] == latents.shape[0]
        assert features.shape[2] == latents.shape[2]

        n_heads = self.heads
        n_batch, n_features, dim = features.shape
        n_queries = latents.shape[1]

        # Layer normalization
        x = self.norm_media(features)
        latents = self.norm_latents(latents)

        # Compute the queries from the latents, for all attention heads simultaneously
        q = self.to_q(latents)
        q = rearrange(q, "b q (h d) -> b h q d", h=n_heads)
        assert q.shape == torch.Size([n_batch, n_heads, n_queries, self.dim_head])

        # Keys and values for all attention heads
        kv_input = torch.cat((x, latents), dim=-2)
        n_features_latents = n_features + n_queries
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        k, v = rearrange_many((k, v), "b f (h d) -> b h f d", h=n_heads)
        assert v.shape == torch.Size(
            [n_batch, n_heads, n_features_latents, self.dim_head]
        )

        q = q * self.scale

        # Attention scores
        sim = einsum("b h q d, b h f d -> b h q f", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        alphas = sim.softmax(dim=-1)

        out = einsum("b h q f, b h f v -> b h q v", alphas, v)
        out = rearrange(out, "b h q v -> b q (h v)")

        return self.to_out(out)


class TempAttentionLayer(nn.Module):
    """Perceiver Attention Layer"""

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        # trainable components of PerceiverAttentionLayer
        self.norm_media = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, features):
        """Latent vectors are cross-attending to the visual features x

        Args:
            features: Batch of visual features with shape (batch_size, n_features, dim)
            latents: Latent learnt vectors which are used to compute queries with shape (batch_size, n_latents, dim)

        Returns:
            Attention score with shape (batch_size, n_latents, dim)
        """
        assert features.ndim == 3

        n_heads = self.heads
        n_batch, n_features, dim = features.shape
        n_queries = features.shape[1]

        # Layer normalization
        x = self.norm_media(features)

        # Compute the queries from the latents, for all attention heads simultaneously
        q = self.to_q(x)
        q = rearrange(q, "b q (h d) -> b h q d", h=n_heads)
        assert q.shape == torch.Size([n_batch, n_heads, n_queries, self.dim_head])

        # Keys and values for all attention heads
        n_features_latents = n_features
        k = self.to_k(x)
        v = self.to_v(x)

        k, v = rearrange_many((k, v), "b f (h d) -> b h f d", h=n_heads)
        assert v.shape == torch.Size(
            [n_batch, n_heads, n_features_latents, self.dim_head]
        )

        q = q * self.scale

        # Attention scores
        sim = einsum("b h q d, b h f d -> b h q f", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        alphas = sim.softmax(dim=-1)

        out = einsum("b h q f, b h f v -> b h q v", alphas, v)
        out = rearrange(out, "b h q v -> b q (h v)")

        return self.to_out(out)


class Video_Former_3D(nn.Module):
    """Perceiver Resampler with multi-head attention layer"""

    def __init__(
        self,
        dim: int,
        depth: int,
        condition_dim: int = 1280,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 64,
        num_frame: int = 12,  #
        num_time_embeds: int = 4,
        ff_mult: int = 4,
        activation: str = "gelu",
        trainable: bool = True,
        use_temporal: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.num_queries = num_latents
        self.num_frame = num_frame
        self.condition_dim = condition_dim
        self.use_temporal = use_temporal

        self.goal_emb = nn.Sequential(
            nn.Linear(condition_dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        frame_seq_len = num_latents // num_frame
        self.latents = nn.Parameter(torch.randn(self.num_frame, frame_seq_len, dim))  # type: ignore[reportPrivateUsage]
        self.time_pos_emb = nn.Parameter(torch.randn(num_time_embeds, 1, dim))  # type: ignore[reportPrivateUsage]
        attn_mask = torch.ones((num_frame, num_frame))
        # attn_mask = torch.tril(attn_mask).bool()

        self.layers = nn.ModuleList([])

        if self.use_temporal:
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            PerceiverAttentionLayer(
                                dim=dim, dim_head=dim_head, heads=heads
                            ),
                            # TempAttentionLayer(dim=dim, dim_head=dim_head, heads=heads),
                            Attention(
                                dim,
                                num_heads=heads,
                                qkv_bias=True,
                                use_cross_attn=False,
                                y_dim=512,
                                attn_mask=attn_mask,
                            ),
                            feed_forward_layer(
                                dim=dim, mult=ff_mult, activation=activation
                            ),
                        ]
                    )
                )
        else:
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            PerceiverAttentionLayer(
                                dim=dim, dim_head=dim_head, heads=heads
                            ),
                            feed_forward_layer(
                                dim=dim, mult=ff_mult, activation=activation
                            ),
                        ]
                    )
                )

        # Layer normalization takes as input the query vector length
        self.norm = nn.LayerNorm(dim)

        self._update_trainable_state(trainable)

    def _update_trainable_state(self, trainable: bool = True):
        for param in self.parameters():
            param.requires_grad = trainable

    def forward(
        self,
        x_f: torch.Tensor,
        mask: torch.BoolTensor = None,
        extra: torch.Tensor = None,
    ):
        """Run perceiver resampler on the input visual embeddings

        Args:
            x_f: Input visual embeddings of shape (batch_size, n_frames, n_features, d_visual)
            mask: Mask for the input visual embeddings of shape (batch_size, n_frames)

        Returns:
            Resampler features of shape (batch_size, num_queries, d_visual)
        """
        assert x_f.ndim == 4

        batch_size, max_length, _, dim = x_f.shape

        # Mask the position embeddings for the padded frames
        time_pos_emb = (
            self.time_pos_emb[:max_length].unsqueeze(0).expand(batch_size, -1, -1, -1)
        )  # [batch_size, max_length, 1, dim]
        if mask is not None:
            time_pos_emb = time_pos_emb * mask.unsqueeze(-1).unsqueeze(-1)

        # Apply the position embeddings
        x_f = self.goal_emb(x_f)
        if extra is not None:
            extra = repeat(extra, "b q d -> b T q d", T=max_length)
            x_f = torch.cat([x_f, extra], dim=2)
        x_f = x_f + time_pos_emb

        # Flatten the frames
        x_f = rearrange(x_f, "b T n d -> (b T) n d")

        # Copy the latents for every element in the batch
        # print(f"debug self.latents.shape {self.latents.shape}")
        x = repeat(self.latents, "T q d -> b T q d", b=batch_size)
        x = rearrange(x, "b T q d -> (b T) q d")
        # print(f"debug: x_f shape {x_f.shape} x shape {x.shape} batch_size {batch_size}")
        # Apply attention and feed forward layer
        if self.use_temporal:
            for attn, Temp_attn, ffw in self.layers:
                x = x + attn(x_f, x)
                x = rearrange(x, "(b T) q d -> (b q) T d", b=batch_size)
                x = x + Temp_attn(x)
                x = rearrange(x, "(b q) T d -> (b T) q d", b=batch_size)
                x = x + ffw(x)
        else:
            for attn, ffw in self.layers:
                x = x + attn(x_f, x)
                x = x + ffw(x)

        # x = rearrange(x, 'l q d -> b T q d', b=batch_size)
        x = x.reshape(batch_size, -1, x.shape[1], x.shape[2])
        x = rearrange(x, "b T q d -> b (T q) d")
        # print(x.shape)
        assert x.shape == torch.Size([batch_size, self.num_queries, self.dim])
        norm = self.norm(x)

        return norm


class Video_Former_2D(nn.Module):
    """Perceiver Resampler with multi-head attention layer"""

    def __init__(
        self,
        dim: int,
        depth: int,
        condition_dim: int = 1280,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 64,
        num_frame: int = 16,
        num_time_embeds: int = 4,
        ff_mult: int = 4,
        activation: str = "gelu",
        trainable: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.num_queries = num_latents
        self.num_frame = num_frame
        self.condition_dim = condition_dim

        self.goal_emb = nn.Sequential(
            nn.Linear(condition_dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        seq_len = num_latents // num_frame
        self.latents = nn.Parameter(torch.randn(num_frame, seq_len, dim))  # type: ignore[reportPrivateUsage]
        self.time_pos_emb = nn.Parameter(torch.randn(num_time_embeds, 1, dim))  # type: ignore[reportPrivateUsage]

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttentionLayer(
                            dim=dim, dim_head=dim_head, heads=heads
                        ),
                        feed_forward_layer(
                            dim=dim, mult=ff_mult, activation=activation
                        ),
                    ]
                )
            )

        # Layer normalization takes as input the query vector length
        self.norm = nn.LayerNorm(dim)

        self._update_trainable_state(trainable)

    def _update_trainable_state(self, trainable: bool = True):
        for param in self.parameters():
            param.requires_grad = trainable

    def forward(self, x_f: torch.Tensor, mask: torch.BoolTensor = None):
        """Run perceiver resampler on the input visual embeddings

        Args:
            x_f: Input visual embeddings of shape (batch_size, n_frames, n_features, d_visual)
            mask: Mask for the input visual embeddings of shape (batch_size, n_frames)

        Returns:
            Resampler features of shape (batch_size, num_queries, d_visual)
        """
        assert x_f.ndim == 4

        batch_size, max_length, _, dim = x_f.shape

        assert dim == self.condition_dim

        # Mask the position embeddings for the padded frames
        time_pos_emb = (
            self.time_pos_emb[:max_length].unsqueeze(0).expand(batch_size, -1, -1, -1)
        )  # [batch_size, max_length, 1, dim]
        if mask is not None:
            time_pos_emb = time_pos_emb * mask.unsqueeze(-1).unsqueeze(-1)

        # Apply the position embeddings
        x_f = self.goal_emb(x_f)
        x_f = x_f + time_pos_emb

        # Flatten the frames
        x_f = rearrange(x_f, "b T n d -> (b T) n d")

        # Copy the latents for every element in the batch
        x = repeat(self.latents, "T q d -> b T q d", b=batch_size)
        x = rearrange(x, "b T q d -> (b T) q d")

        # Apply attention and feed forward layer
        for attn, ffw in self.layers:
            x = x + attn(x_f, x)
            x = x + ffw(x)

        # x = rearrange(x, 'l q d -> b T q d', b=batch_size)
        x = x.reshape(batch_size, -1, x.shape[1], x.shape[2])
        x = rearrange(x, "b T q d -> b (T q) d")
        assert x.shape == torch.Size([batch_size, self.num_queries, self.dim])
        norm = self.norm(x)

        return norm
