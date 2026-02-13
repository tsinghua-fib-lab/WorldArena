import einops

from policy_models.module.transformers.transformer_blocks import *

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        dtype = x.dtype
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device,dtype=dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

logger = logging.getLogger(__name__)

def return_model_parameters_in_millions(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_in_millions = round(num_params / 1_000_000, 2)
    return num_params_in_millions


class DiffusionTransformer(nn.Module):
    """the full GPT score model, with a context size of block_size"""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        device: str,
        n_obs_token: int,
        goal_conditioned: bool,
        action_dim: int,
        proprio_dim: int,
        embed_dim: int,
        embed_pdrob: float,
        attn_pdrop: float,
        resid_pdrop: float,
        mlp_pdrop: float,
        n_dec_layers: int,
        n_enc_layers: int,
        n_heads: int,
        goal_seq_len: int,
        obs_seq_len: int,
        action_seq_len: int,
        goal_drop: float = 0.1,
        bias=False,
        use_mlp_goal: bool = False,
        use_rot_embed: bool = False,
        rotary_xpos: bool = False,
        linear_output: bool = True,
        use_noise_encoder: bool = False,
        use_ada_conditioning: bool = True,
    ):
        super().__init__()
        self.device = device
        self.goal_conditioned = goal_conditioned
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.n_obs_token = n_obs_token
        self.use_ada_conditioning = use_ada_conditioning

        if self.goal_conditioned:
            block_size = goal_seq_len + action_seq_len + obs_seq_len * self.n_obs_token + 2
        else:
            block_size = action_seq_len + obs_seq_len * self.n_obs_token + 2
        self.action_seq_len = action_seq_len
        if self.goal_conditioned:
            seq_size = goal_seq_len + obs_seq_len * self.n_obs_token + action_seq_len
        else:
            seq_size = obs_seq_len * self.n_obs_token + action_seq_len
        print(f"obs dim: {obs_dim}, goal_dim: {goal_dim}, action_dim: {action_dim}, proprio_dim: {proprio_dim}")
        self.tok_emb = nn.Linear(obs_dim, embed_dim)
        if use_mlp_goal:
            self.goal_emb = nn.Sequential(
                nn.Linear(goal_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim)
            )
        else:
            self.goal_emb = nn.Linear(goal_dim, embed_dim)

        if use_mlp_goal:
            self.lang_emb = nn.Sequential(
                nn.Linear(goal_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim)
            )
        else:
            self.lang_emb = nn.Linear(goal_dim, embed_dim)

        if not self.goal_conditioned:
            for param in self.lang_emb.parameters():
                param.requires_grad = False
            for param in self.goal_emb.parameters():
                param.requires_grad = False

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        #print('seq_size:',seq_size)
        self.drop = nn.Dropout(embed_pdrob)
        self.proprio_drop = nn.Dropout(0.5)
        self.cond_mask_prob = goal_drop
        self.use_rot_embed = use_rot_embed
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.latent_encoder_emb = None

        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            n_layers=n_enc_layers,
            block_size=block_size,
            bias=bias,
            use_rot_embed=use_rot_embed,
            rotary_xpos=rotary_xpos,
            mlp_pdrop=mlp_pdrop,
        )

        self.decoder = TransformerFiLMDecoder(
            embed_dim=embed_dim,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            n_layers=n_dec_layers,
            film_cond_dim=embed_dim,
            block_size=block_size,
            bias=bias,
            use_rot_embed=use_rot_embed,
            rotary_xpos=rotary_xpos,
            mlp_pdrop=mlp_pdrop,
            use_cross_attention=True,
            use_noise_encoder=use_noise_encoder,
        )

        self.latent_encoder_emb = None
        self.proprio_emb = nn.Sequential(
            nn.Linear(proprio_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        ).to(self.device)

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len

        self.sigma_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        ).to(self.device).to(torch.bfloat16)

        self.action_emb = nn.Linear(action_dim, embed_dim)

        if linear_output:
            self.action_pred = nn.Linear(embed_dim, self.action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )

        self.apply(self._init_weights)
        logger.info(f'Number of encoder parameters: {return_model_parameters_in_millions(self.encoder)}')
        logger.info(f'Number of decoder parameters: {return_model_parameters_in_millions(self.decoder)}')
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, DiffusionTransformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def forward(self, states, actions, goals, sigma, uncond: Optional[bool] = False):
        context = self.forward_enc_only(states, actions, goals, sigma, uncond)
        pred_actions = self.forward_dec_only(context, actions, sigma)
        # 👇 关键 debug：检查是否可导
        #print(f"[DEBUG] pred_actions.requires_grad: {pred_actions.requires_grad}")
        #print(f"[DEBUG] pred_actions.grad_fn: {pred_actions.grad_fn}")
        #print(f"pred_actions shape: {pred_actions.shape}")
        return pred_actions

    def forward_enc_only(self, states, actions=None, goals=None, sigma=None, uncond: Optional[bool] = False):
        emb_t = self.process_sigma_embeddings(sigma) if not self.use_ada_conditioning else None
        goals = self.preprocess_goals(goals, states['state_images'].size(1), uncond)
        state_embed, proprio_embed = self.process_state_embeddings(states)
        goal_embed = self.process_goal_embeddings(goals)

        input_seq = self.concatenate_inputs(emb_t, goal_embed, state_embed, proprio_embed, uncond)
        context = self.encoder(input_seq)
        self.latent_encoder_emb = context
        return context

    def forward_dec_only(self, context, actions, sigma):
        emb_t = self.process_sigma_embeddings(sigma)
        self.action_emb = self.action_emb.to(emb_t.dtype)
        action_embed = self.action_emb(actions)
        action_x = self.drop(action_embed)
        #print(f"action shape after embedding: {action_x.shape}")
        #print(f"action_x dtype: {action_x.dtype}, emb_t dtype: {emb_t.dtype}, context dtype: {context.dtype}")
        x = self.decoder(action_x, emb_t, context)
        pred_actions = self.action_pred(x)
        return pred_actions

    def process_sigma_embeddings(self, sigma):
        # ====== DEBUG: 打印关键 dtype ======
        #print(f"[DEBUG] sigma input dtype: {sigma.dtype}")
        #print(f"[DEBUG] sigma_emb first layer weight dtype: {self.sigma_emb[0].weight.dtype}")
        #print(f"[DEBUG] model main dtype (first param): {next(self.parameters()).dtype}")
        # ===================================
        sigmas = sigma.log() / 4
        sigmas = einops.rearrange(sigmas, 'b -> b 1')
        self.sigma_emb = self.sigma_emb.to(sigmas.dtype)
        emb_t = self.sigma_emb(sigmas)
        if len(emb_t.shape) == 2:
            emb_t = einops.rearrange(emb_t, 'b d -> b 1 d')
        return emb_t

    def preprocess_goals(self, goals, states_length, uncond=False):
        if len(goals.shape) == 2:
            goals = einops.rearrange(goals, 'b d -> b 1 d')
        if goals.shape[1] == states_length and self.goal_seq_len == 1:
            goals = goals[:, 0, :]
            goals = einops.rearrange(goals, 'b d -> b 1 d')
        if goals.shape[-1] == 2 * self.obs_dim:
            goals = goals[:, :, :self.obs_dim]
        if self.training:
            goals = self.mask_cond(goals)
        if uncond:
            goals = torch.zeros_like(goals).to(self.device)
        return goals

    def process_state_embeddings(self, states):
        states_global = self.tok_emb(states['state_images'])
        if 'state_obs' in states:
            proprio_embed = self.proprio_emb(states['state_obs'])
        else:
            proprio_embed = None
        return states_global, proprio_embed

    def process_goal_embeddings(self, goals):
        goal_embed = self.lang_emb(goals)
        return goal_embed

    def apply_position_embeddings(self, goal_embed, state_embed, action_embed, proprio_embed, t):
        pos_len = t + self.goal_seq_len + self.action_seq_len - 1
        position_embeddings = self.pos_emb[:, :pos_len, :]
        goal_x = self.drop(goal_embed + position_embeddings[:, :self.goal_seq_len, :])
        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])
        action_x = self.drop(action_embed + position_embeddings[:, (self.goal_seq_len + t - 1):, :])
        proprio_x = self.drop(proprio_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :]) if proprio_embed is not None else None
        return goal_x, state_x, action_x, proprio_x

    def concatenate_inputs(self, emb_t, goal_x, state_x, proprio_x, uncond=False):
        input_seq_components = [state_x]

        if self.goal_conditioned:
            input_seq_components.insert(0, goal_x)

        if proprio_x is not None:
            input_seq_components.append(proprio_x)
        #else:
        #    if not self.goal_conditioned:
        #        input_seq_components.append(self.drop(goal_x))

        input_seq = torch.cat(input_seq_components, dim=1)
        return input_seq

    def mask_cond(self, cond, force_mask=False):
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones((bs, t, d), device=cond.device) * self.cond_mask_prob)
            return cond * (1. - mask)
        else:
            return cond

    def get_params(self):
        return self.parameters()