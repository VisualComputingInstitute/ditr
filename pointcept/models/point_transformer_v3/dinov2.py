from typing import Literal

import timm
import torch
import torch.nn as nn
from einops import rearrange


@torch.compile(fullgraph=True)
class DINOv2(nn.Module):
    def __init__(
        self,
        model: Literal["small", "base", "large", "giant"] | str = "base",
    ):
        super().__init__()
        model = (
            f"vit_{model}_patch14_reg4_dinov2"
            if model in ["small", "base", "large", "giant"]
            else model  # e.g. "vit_large_patch16_384.augreg_in21k_ft_in1k"
        )
        self.vit = timm.create_model(
            model,
            pretrained=True,
            dynamic_img_size=True,
            num_classes=0,
            global_pool="",
        )

    def forward(
        self,
        images: torch.Tensor,  # (B, C, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        H_tok, W_tok = self.vit.patch_embed.dynamic_feat_size(images.shape[-2:])
        vit_output = self.vit(images)
        cls_token = vit_output[:, 0]
        feat_map = rearrange(
            vit_output[:, self.vit.num_prefix_tokens :],
            "b (h w) c -> b c h w",
            h=H_tok,
            w=W_tok,
        )

        return feat_map, cls_token

    @property
    def output_channels(self) -> int:
        return self.vit.embed_dim

    @property
    def patch_size(self) -> int:
        return self.vit.patch_embed.patch_size[0]


@torch.compile(fullgraph=True)
class FrozenDINOv2:  # not an nn.Module, keeps the checkpoint file small
    def __init__(
        self,
        model: Literal["small", "base", "large", "giant"] | str = "base",
    ):
        super().__init__()
        model = (
            f"vit_{model}_patch14_reg4_dinov2"
            if model in ["small", "base", "large", "giant"]
            else model  # e.g. "vit_large_patch16_384.augreg_in21k_ft_in1k"
        )
        self.vit = timm.create_model(
            model,
            pretrained=True,
            dynamic_img_size=True,
            num_classes=0,
            global_pool="",
        ).cuda()  # need to manually move to GPU; it's not an nn.Module
        self.vit.requires_grad_(False)
        self.vit.eval()

    @torch.no_grad()  # shouldn't be necessary, but just to be safe
    def __call__(
        self,
        images: torch.Tensor,  # (B, C, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert not self.vit.training
        H_tok, W_tok = self.vit.patch_embed.dynamic_feat_size(images.shape[-2:])
        vit_output = self.vit(images)
        cls_token = vit_output[:, 0]
        feat_map = rearrange(
            vit_output[:, self.vit.num_prefix_tokens :],
            "b (h w) c -> b c h w",
            h=H_tok,
            w=W_tok,
        )

        return feat_map, cls_token

    @property
    def output_channels(self) -> int:
        return self.vit.embed_dim

    @property
    def patch_size(self) -> int:
        return self.vit.patch_embed.patch_size[0]
