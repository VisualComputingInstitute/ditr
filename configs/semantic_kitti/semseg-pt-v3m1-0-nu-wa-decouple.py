_base_ = ["semseg-pt-v3m1-0-base.py"]

find_unused_parameters = True

# model settings
model = dict(
    backbone=dict(
        pdnorm_bn=True,
        pdnorm_ln=True,
        pdnorm_conditions=("nuScenes", "Waymo"),
    ),
)

# dataset settings
data = dict(
    train=dict(
        transform=[
            # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
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
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            dict(type="Add", keys_dict={"condition": "Waymo"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "condition"),
                feat_keys=("coord", "strength"),
            ),
        ],
    ),
    val=dict(
        transform=[
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            dict(type="Add", keys_dict={"condition": "Waymo"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "condition"),
                feat_keys=("coord", "strength"),
            ),
        ],
    ),
    test=dict(
        test_cfg=dict(
            post_transform=[
                dict(type="Add", keys_dict={"condition": "Waymo"}),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "condition"),
                    feat_keys=("coord", "strength"),
                ),
            ],
        ),
    ),
)
