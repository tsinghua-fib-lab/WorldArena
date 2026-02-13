from typing import List

import torch
import torch.nn as nn

from policy_models.module.clip import build_model, load_clip, tokenize


class LangClip(nn.Module):
    def __init__(self, freeze_backbone: bool = True, model_name: str = "RN50"):
        super(LangClip, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load CLIP model
        print(f"loading language CLIP model with backbone: {model_name}")
        self._load_clip(model_name)
        if freeze_backbone:
            for param in self.clip_rn50.parameters():
                param.requires_grad = False

    def _load_clip(self, model_name: str) -> None:
        model, _ = load_clip(model_name, device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)

    def forward(self, x: List) -> torch.Tensor:
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            emb = self.clip_rn50.encode_text(tokens)
        return torch.unsqueeze(emb, 1)
