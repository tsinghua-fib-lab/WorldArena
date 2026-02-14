from torch import nn
from .utils import append_dims
from policy_models.module.diffusion_decoder import DiffusionTransformer

'''
Wrappers for the score-based models based on Karras et al. 2022
They are used to get improved scaling of different noise levels, which
improves training stability and model performance 

Code is adapted from:

https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/layers.py
'''


class GCDenoiser(nn.Module):
    """
    A Karras et al. preconditioner for denoising diffusion models.

    Args:
        inner_model: The inner model used for denoising.
        sigma_data: The data sigma for scalings (default: 1.0).
    """
    def __init__(self, action_dim, obs_dim, goal_dim, num_tokens, goal_window_size, obs_seq_len, act_seq_len, device, sigma_data=1., proprio_dim=8):
        super().__init__()
        self.inner_model = DiffusionTransformer(
            action_dim = action_dim,
            obs_dim = obs_dim,
            goal_dim = goal_dim,
            proprio_dim= proprio_dim,
            goal_conditioned = True,
            embed_dim = 384,
            n_dec_layers = 4,
            n_enc_layers = 4,
            n_obs_token = num_tokens,
            goal_seq_len = goal_window_size,
            obs_seq_len = obs_seq_len,
            action_seq_len =act_seq_len,
            embed_pdrob = 0,
            goal_drop = 0,
            attn_pdrop = 0.3,
            resid_pdrop = 0.1,
            mlp_pdrop = 0.05,
            n_heads= 8,
            device = device,
            use_mlp_goal = True,
        )
        self.sigma_data = sigma_data

    def get_scalings(self, sigma):
        """
        Compute the scalings for the denoising process.

        Args:
            sigma: The input sigma.
        Returns:
            The computed scalings for skip connections, output, and input.
        """
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def loss(self, state, action, goal, noise, sigma, **kwargs):
        """
        Compute the loss for the denoising process.

        Args:
            state: The input state.
            action: The input action.
            goal: The input goal.
            noise: The input noise.
            sigma: The input sigma.
            **kwargs: Additional keyword arguments.
        Returns:
            The computed loss.
        """
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        noised_input = action + noise * append_dims(sigma, action.ndim)
        model_output = self.inner_model(state, noised_input * c_in, goal, sigma, **kwargs)
        #print(f"[DEBUG] model_output.requires_grad: {model_output.requires_grad}")
        #print(f"[DEBUG] model_output.grad_fn: {model_output.grad_fn}")
        target = (action - c_skip * noised_input) / c_out
        return (model_output - target).pow(2).flatten(1).mean(), model_output

    def forward(self, state, action, goal, sigma, **kwargs):
        """
        Perform the forward pass of the denoising process.

        Args:
            state: The input state.
            action: The input action.
            goal: The input goal.
            sigma: The input sigma.
            **kwargs: Additional keyword arguments.

        Returns:
            The output of the forward pass.
        """
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        return self.inner_model(state, action * c_in, goal, sigma, **kwargs) * c_out + action * c_skip
    
    def forward_context_only(self, state, action, goal, sigma, **kwargs):
        """
        Perform the forward pass of the denoising process.

        Args:
            state: The input state.
            action: The input action.
            goal: The input goal.
            sigma: The input sigma.
            **kwargs: Additional keyword arguments.

        Returns:
            The output of the forward pass.
        """
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        return self.inner_model.forward_enc_only(state, action * c_in, goal, sigma, **kwargs)

    def get_params(self):
        return self.inner_model.parameters()
