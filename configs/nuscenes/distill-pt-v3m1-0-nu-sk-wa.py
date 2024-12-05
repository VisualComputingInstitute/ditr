_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 12  # bs: total bs in all gpus
num_worker = 24
mix_prob = 0.8
empty_cache = False
enable_amp = True
# find_unused_parameters = True
wandb_project = "pointcept_distill_nuscenes"

# trainer
train = dict(
    type="MultiDatasetTrainer",
)

# model settings
model = dict(
    type="DefaultDistiller",
    backbone_out_channels=64,
    dinov2="large",
    with_cls_tokens=None,
    backbone_bottleneck_channels=512,
    backbone=dict(
        type="PT-v3m1",
        in_channels=4,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
    ),
    criteria=[
        dict(type="CosineLoss", loss_weight=1.0),
    ],
    conditions=("nuScenes", "SemanticKITTI", "Waymo"),
    head_decouple=False,
)

# scheduler settings
epoch = 50
eval_epoch = 5
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword=r"img_enc|block", lr=0.0002)]

# dataset settings
data = dict(
    num_classes=16,
    ignore_index=-1,
    names=[
        "barrier",
        "bicycle",
        "bus",
        "car",
        "construction_vehicle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "trailer",
        "truck",
        "driveable_surface",
        "other_flat",
        "sidewalk",
        "terrain",
        "manmade",
        "vegetation",
    ],
    train=dict(
        type="ConcatDataset",
        datasets=[
            # nuScenes
            dict(
                type="NuScenesDataset",
                split="train",
                data_root="data/nuscenes",
                with_images=True,
                transform=[
                    dict(type="ImageResize", size=[378, 672]),
                    dict(
                        type="ImageColorJitter",
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.1,
                    ),
                    dict(type="ImageRandomHorizontalFlip"),
                    dict(type="ImageNormalize"),
                    # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        keys=(
                            "coord",
                            "strength",
                            "segment",
                            "image_coord",
                            "image_mask",
                        ),
                        return_grid_coord=True,
                    ),
                    # dict(type="SphereCrop", point_max=1000000, mode="random"),
                    # dict(type="CenterShift", apply_z=False),
                    dict(type="Add", keys_dict={"condition": "nuScenes"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=(
                            "coord",
                            "grid_coord",
                            "segment",
                            "condition",
                            "image",
                            "image_coord",
                            "image_mask",
                        ),
                        feat_keys=("coord", "strength"),
                    ),
                ],
                test_mode=False,
                ignore_index=-1,
                loop=1,
            ),
            # SemanticKITTI
            dict(
                type="SemanticKITTIDataset",
                split="train",
                data_root="data/semantic_kitti",
                with_images=True,
                transform=[
                    dict(type="ImageResize", size=[378, 1246]),
                    dict(
                        type="ImageColorJitter",
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.1,
                    ),
                    dict(type="ImageRandomHorizontalFlip"),
                    dict(type="ImageNormalize"),
                    # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
                    dict(
                        type="PointClip",
                        point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2),
                    ),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        keys=(
                            "coord",
                            "strength",
                            "segment",
                            "image_coord",
                            "image_mask",
                        ),
                        return_grid_coord=True,
                    ),
                    # dict(type="SphereCrop", point_max=1000000, mode="random"),
                    # dict(type="CenterShift", apply_z=False),
                    dict(type="Add", keys_dict={"condition": "SemanticKITTI"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=(
                            "coord",
                            "grid_coord",
                            "segment",
                            "condition",
                            "image",
                            "image_coord",
                            "image_mask",
                        ),
                        feat_keys=("coord", "strength"),
                    ),
                ],
                test_mode=False,
                ignore_index=-1,
                loop=1,
            ),
            # Waymo
            dict(
                type="WaymoDataset",
                split="training",
                data_root="data/waymo",
                with_images=True,
                transform=[
                    dict(type="ImageResize", size=[308, 672]),
                    dict(
                        type="ImageColorJitter",
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.1,
                    ),
                    dict(type="ImageRandomHorizontalFlip"),
                    dict(type="ImageNormalize"),
                    # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
                    dict(
                        type="PointClip",
                        point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2),
                    ),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        keys=(
                            "coord",
                            "strength",
                            "segment",
                            "image_coord",
                            "image_mask",
                        ),
                        return_grid_coord=True,
                    ),
                    # dict(type="SphereCrop", point_max=1000000, mode="random"),
                    # dict(type="CenterShift", apply_z=False),
                    dict(type="Add", keys_dict={"condition": "Waymo"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=(
                            "coord",
                            "grid_coord",
                            "segment",
                            "condition",
                            "image",
                            "image_coord",
                            "image_mask",
                        ),
                        feat_keys=("coord", "strength"),
                    ),
                ],
                test_mode=False,
                ignore_index=-1,
                loop=1,
            ),
        ],
    ),
    val=dict(
        type="NuScenesDataset",
        split="val",
        data_root="data/nuscenes",
        with_images=True,
        transform=[
            dict(type="ImageResize", size=[378, 672]),
            dict(type="ImageNormalize"),
            # dict(type="PointClip", point_cloud_range=(-51.2, -51.2, -4, 51.2, 51.2, 2.4)),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment", "image_coord", "image_mask"),
                return_grid_coord=True,
            ),
            dict(type="Add", keys_dict={"condition": "nuScenes"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "condition",
                    "image",
                    "image_coord",
                    "image_mask",
                ),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=-1,
    ),
    test=dict(
        type="NuScenesDataset",
        split="val",
        data_root="data/nuscenes",
        with_images=True,
        transform=[
            dict(type="ImageResize", size=[378, 672]),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.025,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment", "image_coord", "image_mask"),
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                keys=("coord", "strength", "image_coord", "image_mask"),
            ),
            crop=None,
            post_transform=[
                dict(type="ImageNormalize"),
                dict(type="Add", keys_dict={"condition": "nuScenes"}),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=(
                        "coord",
                        "grid_coord",
                        "index",
                        "condition",
                        "image",
                        "image_coord",
                        "image_mask",
                    ),
                    feat_keys=("coord", "strength"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
        ),
        ignore_index=-1,
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="LossEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]

test = dict(_delete_=True)
