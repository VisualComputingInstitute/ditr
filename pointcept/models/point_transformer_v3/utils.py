import torch
from einops import rearrange


def get_image_feat(img_enc, data_dict) -> tuple[torch.Tensor, torch.Tensor, int]:
    feat_map, cls_token = img_enc(
        rearrange(data_dict["image"], "b cam c h w -> (b cam) c h w")
    )
    feat_map = rearrange(
        feat_map,
        "(b cam) c h w -> b cam c h w",
        b=data_dict["image"].shape[0],
    )
    cls_token = rearrange(
        cls_token,
        "(b cam) c -> b cam c",
        b=data_dict["image"].shape[0],
    )
    return feat_map, cls_token, img_enc.patch_size


def assign_image_feat(
    data_dict: dict,
    image_feat: torch.Tensor,  # (B, CAM, C, H, W)
    patch_size: int,
    missing_feature_embedding: torch.Tensor | None = None,  # (C,)
) -> torch.Tensor:  # (N, C)
    """
    data_dict["image_coord"]: (N, CAM, 2)
    data_dict["image_mask"]: (N, CAM)
    data_dict["offset"]: (B,)
    """

    point_image_feat = []

    offset = (
        data_dict["unmix3d_offset"]
        if "unmix3d_offset" in data_dict.keys()
        else data_dict["offset"]
    ).tolist()
    splits = [o - offset[i - 1] if i > 0 else o for i, o in enumerate(offset)]
    image_coord = torch.split(
        data_dict["image_coord"] / patch_size, splits
    )  # [(N_i, CAM, 2)]
    image_mask = torch.split(data_dict["image_mask"], splits)  # [(N_i, CAM, 2)]

    for i, image_feat_i in enumerate(image_feat):
        N_i = image_coord[i].shape[0]
        C = image_feat_i.shape[1]
        point_image_feat_i = (
            missing_feature_embedding.expand(N_i, -1).clone()
            if missing_feature_embedding is not None
            else image_feat_i.new_zeros(N_i, C)
        )

        for j, image_feat_ij in enumerate(image_feat_i):
            image_coord_ij = image_coord[i][:, j]
            image_mask_ij = image_mask[i][:, j]

            # temporary fix; should be handled a bit more elegantly. after augmentations
            # and scaling, the coords can be out of bounds due to rounding errors.
            H, W = image_feat_ij.shape[-2:]
            image_mask_ij = torch.logical_and(image_mask_ij, image_coord_ij[:, 1] < H)
            image_mask_ij = torch.logical_and(image_mask_ij, image_coord_ij[:, 0] < W)

            point_image_feat_i[image_mask_ij] = rearrange(
                image_feat_ij, "c h w -> h w c"
            )[
                image_coord_ij[image_mask_ij][:, 1].int(),
                image_coord_ij[image_mask_ij][:, 0].int(),
            ]

        point_image_feat.append(point_image_feat_i)

    return torch.cat(point_image_feat)


def mix3d_cls_token(
    data_dict: dict,
    cls_token: torch.Tensor,  # (B, CAM, C)
) -> torch.Tensor:  # (B', CAM, C)
    """
    reshapes unmix3d cls tokens to mix3d cls tokens if necessary
    """
    if "unmix3d_offset" in data_dict.keys():
        assert len(data_dict["unmix3d_offset"]) / len(data_dict["offset"]) == 2
        return rearrange(cls_token, "(b mix) cam c -> b (mix cam) c", mix=2)
    else:
        return cls_token
