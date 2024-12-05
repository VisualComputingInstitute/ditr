_base_ = ["semseg-pt-v3m1-0-image.py"]

model = dict(
    backbone=dict(
        type="PT-v3m1-image-self-attn",
        enc_image_cls_token_attn=[],
        dec_image_cls_token_attn=[0, 1, 2, 3],
    ),
)
