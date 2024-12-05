"""
nuScenes Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Zheng Zhang
Please cite our work if the code is helpful to you.
"""

import os
import pickle
from collections.abc import Sequence

import imageio.v3 as imageio
import numpy as np
import torch

from pointcept.datasets.utils import points2image

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class NuScenesDataset(DefaultDataset):
    def __init__(self, sweeps=10, ignore_index=-1, with_images=False, **kwargs):
        self.sweeps = sweeps
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.with_images = with_images
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_info_path(self, split):
        assert split in ["train", "val", "test"]
        if split == "train":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_train.pkl"
            )
        elif split == "val":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_val.pkl"
            )
        elif split == "test":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_test.pkl"
            )
        else:
            raise NotImplementedError

    def get_data_list(self):
        if isinstance(self.split, str):
            info_paths = [self.get_info_path(self.split)]
        elif isinstance(self.split, Sequence):
            info_paths = [self.get_info_path(s) for s in self.split]
        else:
            raise NotImplementedError
        data_list = []
        for info_path in info_paths:
            with open(info_path, "rb") as f:
                info = pickle.load(f)
                data_list.extend(info)
        return data_list

    def get_data(self, idx):
        data = self.data_list[idx % len(self.data_list)]
        lidar_path = os.path.join(self.data_root, "raw", data["lidar_path"])
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape(
            [-1, 5]
        )
        coord = points[:, :3]
        strength = points[:, 3].reshape([-1, 1]) / 255  # scale strength to [0, 1]

        if "gt_segment_path" in data.keys():
            gt_segment_path = os.path.join(
                self.data_root, "raw", data["gt_segment_path"]
            )
            segment = np.fromfile(
                str(gt_segment_path), dtype=np.uint8, count=-1
            ).reshape([-1])
            segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(
                np.int64
            )
        else:
            segment = np.ones((points.shape[0],), dtype=np.int64) * self.ignore_index
        data_dict = dict(
            coord=coord,
            strength=strength,
            segment=segment,
            name=self.get_data_name(idx),
        )

        if self.with_images:
            images = []
            image_coords = []
            image_masks = []
            for cam in data["cams"].values():
                image_path = os.path.join(self.data_root, "raw", cam["data_path"])
                images.append(imageio.imread(image_path))
                K = cam["camera_intrinsics"]
                cam2points = np.eye(4)
                cam2points[:3, :3] = cam["sensor2lidar_rotation"]
                cam2points[:3, 3] = cam["sensor2lidar_translation"]
                points2cam = np.linalg.inv(cam2points)

                points_image, visibility_mask = points2image(
                    coord, points2cam, K, images[-1].shape[:2]
                )
                image_coords.append(points_image)
                image_masks.append(visibility_mask)

            data_dict["image"] = np.stack(images, axis=0)  # (CAM, H, W, 3)
            data_dict["image_coord"] = np.stack(image_coords, axis=1)  # (N, CAM, 2)
            data_dict["image_mask"] = np.stack(image_masks, axis=1)  # (N, CAM)

        return data_dict

    def get_data_name(self, idx):
        # return data name for lidar seg, optimize the code when need to support detection
        return self.data_list[idx % len(self.data_list)]["lidar_token"]

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,
            1: ignore_index,
            2: 6,
            3: 6,
            4: 6,
            5: ignore_index,
            6: 6,
            7: ignore_index,
            8: ignore_index,
            9: 0,
            10: ignore_index,
            11: ignore_index,
            12: 7,
            13: ignore_index,
            14: 1,
            15: 2,
            16: 2,
            17: 3,
            18: 4,
            19: ignore_index,
            20: ignore_index,
            21: 5,
            22: 8,
            23: 9,
            24: 10,
            25: 11,
            26: 12,
            27: 13,
            28: 14,
            29: ignore_index,
            30: 15,
            31: ignore_index,
        }
        return learning_map
