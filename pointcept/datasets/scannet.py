"""
ScanNet20 / ScanNet200 / ScanNet Data Efficient Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import glob
import os

import imageio.v3 as imageio
import numpy as np
import torch

from pointcept.datasets.utils import points2image
from pointcept.utils.cache import shared_dict

from .builder import DATASETS
from .defaults import DefaultDataset
from .preprocessing.scannet.meta_data.scannet200_constants import (
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
)


@DATASETS.register_module()
class ScanNetDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment20",
        "instance",
    ]
    class2id = np.array(VALID_CLASS_IDS_20)

    def __init__(
        self,
        lr_file=None,
        la_file=None,
        with_images=False,
        **kwargs,
    ):
        self.lr = np.loadtxt(lr_file, dtype=str) if lr_file is not None else None
        self.la = torch.load(la_file) if la_file is not None else None
        self.with_images = with_images
        super().__init__(**kwargs)

    def get_data_list(self):
        if self.lr is None:
            data_list = super().get_data_list()
        else:
            data_list = [
                os.path.join(self.data_root, "train", name) for name in self.lr
            ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["coord"] = data_dict["coord"].astype(np.float32)
        data_dict["color"] = data_dict["color"].astype(np.float32)
        data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment20" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment20").reshape([-1]).astype(np.int32)
            )
        elif "segment200" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment200").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if "instance" in data_dict.keys():
            data_dict["instance"] = (
                data_dict.pop("instance").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            mask = np.ones_like(data_dict["segment"], dtype=bool)
            mask[sampled_index] = False
            data_dict["segment"][mask] = self.ignore_index
            data_dict["sampled_index"] = sampled_index

        if self.with_images:
            images = []
            image_coords = []
            image_masks = []
            image_path = f"data/scannet_images/{name}/color/"
            intrinsic_path = f"data/scannet_images/{name}/intrinsic/intrinsic_color.txt"
            intrinsic = np.loadtxt(intrinsic_path)[:3, :3]

            all_images = glob.glob(os.path.join(image_path, "*.jpg"))
            if self.split == "val" or self.split == "test":
                step = len(all_images) // self.with_images
                selected_indices = [i * step for i in range(self.with_images)]
            else:
                selected_indices = np.random.choice(
                    len(all_images), self.with_images, replace=False
                )
            selected_images = [os.path.join(image_path, f"{i*10}.jpg") for i in selected_indices]
            for img_path in selected_images:
                image = imageio.imread(img_path)
                H, W = image.shape[:2]
                images.append(image)

                depth_path = os.path.join(
                    os.path.dirname(img_path).replace("color", "depth"),
                    os.path.basename(img_path).replace(".jpg", ".png"),
                )
                depth = imageio.imread(depth_path) / 1000

                extrinsic_path = os.path.join(
                    os.path.dirname(img_path).replace("color", "pose"),
                    os.path.basename(img_path).replace(".jpg", ".txt"),
                )
                points2cam = np.linalg.inv(np.loadtxt(extrinsic_path))

                points_image, visibility_mask = points2image(
                    data_dict["coord"],
                    points2cam,
                    intrinsic,
                    (H, W),
                    depth=depth,
                    min_distance=1.0,
                    max_distance=4.0,
                    error_margin=0.2,
                )
                image_coords.append(points_image)
                image_masks.append(visibility_mask)
            data_dict["image"] = np.stack(images, axis=0)  # (CAM, H, W, 3)
            data_dict["image_coord"] = np.stack(image_coords, axis=1)  # (N, CAM, 2)
            data_dict["image_mask"] = np.stack(image_masks, axis=1)  # (N, CAM)

        return data_dict


@DATASETS.register_module()
class ScanNet200Dataset(ScanNetDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment200",
        "instance",
    ]
    class2id = np.array(VALID_CLASS_IDS_200)
