import torch
import torch.nn as nn
import torch_scatter
from einops import rearrange

from pointcept.models.losses import build_criteria
from pointcept.models.point_transformer_v3.dinov2 import FrozenDINOv2
from pointcept.models.point_transformer_v3.utils import (
    assign_image_feat,
    get_image_feat,
    mix3d_cls_token,
)
from pointcept.models.utils.structure import Point

from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultDistiller(nn.Module):
    def __init__(
        self,
        backbone_out_channels,
        dinov2,
        backbone_bottleneck_channels=None,
        with_cls_tokens=None,
        backbone=None,
        criteria=None,
        cls_token_criteria=None,
        context_channels=None,
        conditions=None,
        head_decouple=False,
    ):
        super().__init__()
        self.img_enc = FrozenDINOv2(model=dinov2)

        if head_decouple:
            assert conditions is not None
            self.proj = nn.ModuleList(
                nn.Linear(backbone_out_channels, self.img_enc.output_channels)
                for _ in range(len(conditions))
            )
        else:
            self.proj = nn.Linear(backbone_out_channels, self.img_enc.output_channels)

        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        if with_cls_tokens is not None:
            assert cls_token_criteria is not None
            assert backbone_bottleneck_channels is not None
            self.cls_token_criteria = build_criteria(cls_token_criteria)
            if head_decouple:
                assert conditions is not None
                self.proj_cls_token = nn.ModuleList(
                    nn.Linear(
                        backbone_bottleneck_channels,
                        self.img_enc.output_channels * with_cls_tokens[i],
                    )
                    for i in range(len(conditions))
                )
            else:
                self.proj_cls_token = nn.Linear(
                    backbone_bottleneck_channels,
                    self.img_enc.output_channels * with_cls_tokens,
                )
        self.with_cls_tokens = with_cls_tokens
        self.conditions = conditions
        if conditions is not None and context_channels is not None:
            self.embedding_table = nn.Embedding(len(conditions), context_channels)
        self.head_decouple = head_decouple

    def forward(self, input_dict):
        proj = self.proj
        if self.with_cls_tokens:
            proj_cls_token = self.proj_cls_token
        if self.conditions is not None:
            condition = input_dict["condition"][0]
            assert condition in self.conditions

            if hasattr(self, "embedding_table"):
                context = self.embedding_table(
                    torch.tensor(
                        [self.conditions.index(condition)],
                        device=input_dict["coord"].device,
                    )
                )
                input_dict["context"] = context

            if self.head_decouple:
                proj = self.proj[self.conditions.index(condition)]
                if self.with_cls_tokens:
                    proj_cls_token = self.proj_cls_token[
                        self.conditions.index(condition)
                    ]
        else:
            assert "condition" not in input_dict.keys()

        image_feat, image_cls_token, patch_size = get_image_feat(
            self.img_enc, input_dict
        )  # (B, CAM, C, H, W)
        image_feat = assign_image_feat(input_dict, image_feat, patch_size)  # (N, C)
        if self.with_cls_tokens:
            image_cls_token = mix3d_cls_token(
                input_dict, image_cls_token
            )  # (B, CAM, C)

        point = Point(input_dict)
        point = self.backbone(point)

        pred = proj(point.feat)
        target = image_feat
        if "image_mask" in point.keys():
            msk = point.image_mask.any(dim=1)  # visible in any view
            pred = pred[msk]
            target = target[msk]

        out_dict = dict()
        out_dict["patch_loss"] = self.criteria(pred, target)
        out_dict["loss"] = out_dict["patch_loss"]

        if self.with_cls_tokens:
            bottleneck = point
            while "unpooling_parent" in bottleneck.keys():
                bottleneck = bottleneck.unpooling_parent

            pooled_feat = torch_scatter.segment_csr(
                src=bottleneck.feat,
                indptr=nn.functional.pad(bottleneck.offset, (1, 0)),
                reduce="mean",
            )

            out_dict["cls_token_loss"] = self.cls_token_criteria(
                rearrange(
                    proj_cls_token(pooled_feat),
                    "b (cam c) -> (b cam) c",
                    cam=image_cls_token.shape[1],
                    c=image_cls_token.shape[2],
                ),
                rearrange(image_cls_token, "b cam c -> (b cam) c"),
            )
            out_dict["loss"] = (
                out_dict["loss"] + out_dict["cls_token_loss"]
            )  # don't use +=

        if self.training:
            return out_dict

        out_dict["feat"] = point.feat
        return out_dict


@MODELS.register_module()
class TeacherStudent(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        levels,
        backbone=None,
        teacher=None,
        ckpt_path=None,
        load_student=False,
        distill_criteria=None,
        seg_criteria=None,
    ):
        super().__init__()
        if seg_criteria is not None:
            self.seg_head = (
                nn.Linear(backbone_out_channels, num_classes)
                if num_classes > 0
                else nn.Identity()
            )
        self.backbone = build_model(backbone)
        self.teacher = build_model(teacher)
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path)["state_dict"]
            if seg_criteria is not None:
                self.seg_head.weight.data = state_dict["module.seg_head.weight"]
                self.seg_head.bias.data = state_dict["module.seg_head.bias"]

            weight = dict()
            for key, value in state_dict.items():
                if key.startswith("module.backbone"):
                    key = key.replace("module.backbone.", "").replace(
                        "context_qkv", "qkv_cls_token"
                    )
                    weight[key] = value
            self.teacher.load_state_dict(weight, strict=True)
            if load_student:
                weight = {
                    key: value
                    for key, value in weight.items()
                    if ("proj_image_feat" not in key) and ("qkv_cls_token" not in key)
                }
                self.backbone.load_state_dict(weight, strict=True)

        self.distill_criteria = build_criteria(distill_criteria)
        self.seg_criteria = (
            build_criteria(seg_criteria) if seg_criteria is not None else None
        )
        self.levels = levels
        self.distill_decay = 1.0

    def train(self, mode: bool = True):
        super().train(mode)

        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def forward(self, input_dict):
        backbone_point = Point(input_dict)
        teacher_point = Point(input_dict)
        backbone_point = self.backbone(backbone_point)
        teacher_point = self.teacher(teacher_point)

        if self.seg_criteria is not None:
            feat = backbone_point.feat
            seg_logits = self.seg_head(feat)

        backbone_points = [backbone_point]
        while "unpooling_parent" in backbone_points[-1].keys():
            backbone_points.append(backbone_points[-1].unpooling_parent)

        teacher_points = [teacher_point]
        while "unpooling_parent" in teacher_points[-1].keys():
            teacher_points.append(teacher_points[-1].unpooling_parent)

        out_dict = dict()
        if self.training:
            distill_losses = []
            for lvl in self.levels:
                pred = backbone_points[lvl]
                target = teacher_points[lvl]
                distill_losses.append(self.distill_criteria(pred.feat, target.feat))
            for i, loss in enumerate(distill_losses):
                out_dict[f"distill_loss_lvl_{self.levels[i]}"] = loss

            out_dict["distill_decay"] = torch.tensor(self.distill_decay)
            out_dict["distill_loss"] = (
                self.distill_decay * sum(distill_losses) / len(distill_losses)
            )

            if self.seg_criteria is not None:
                out_dict["seg_loss"] = self.seg_criteria(
                    seg_logits, input_dict["segment"]
                )
                out_dict["loss"] = out_dict["seg_loss"] + out_dict["distill_loss"]
            else:
                out_dict["loss"] = out_dict["distill_loss"]
        elif "segment" in input_dict.keys():
            distill_losses = []
            for lvl in self.levels:
                pred = backbone_points[lvl]
                target = teacher_points[lvl]
                distill_losses.append(self.distill_criteria(pred.feat, target.feat))
            for i, loss in enumerate(distill_losses):
                out_dict[f"distill_loss_lvl_{self.levels[i]}"] = loss

            out_dict["distill_loss"] = (
                self.distill_decay * sum(distill_losses) / len(distill_losses)
            )

            if self.seg_criteria is not None:
                out_dict["seg_loss"] = self.seg_criteria(
                    seg_logits, input_dict["segment"]
                )
                out_dict["loss"] = out_dict["seg_loss"] + out_dict["distill_loss"]
                out_dict["seg_logits"] = seg_logits
            else:
                out_dict["loss"] = out_dict["distill_loss"]
        else:
            if self.seg_criteria is not None:
                out_dict["seg_logits"] = seg_logits

        return out_dict


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
