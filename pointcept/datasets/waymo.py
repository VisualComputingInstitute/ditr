"""
Waymo dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import glob
import os

import imageio.v3 as imageio
import numpy as np
import torch
from einops import repeat

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class WaymoDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "strength",
        "segment",
        "instance",
        "pose",
        "image_coord",
        "image_mask",
    ]

    def __init__(
        self,
        timestamp=(0,),
        reference_label=True,
        timing_embedding=False,
        with_images=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert timestamp[0] == 0
        self.timestamp = timestamp
        self.reference_label = reference_label
        self.timing_embedding = timing_embedding
        self.data_list = sorted(self.data_list)
        _, self.sequence_offset, self.sequence_index = np.unique(
            [os.path.dirname(data) for data in self.data_list],
            return_index=True,
            return_inverse=True,
        )
        self.sequence_offset = np.append(self.sequence_offset, len(self.data_list))
        self.with_images = with_images

    def get_data_list(self):
        if isinstance(self.split, str):
            self.split = [self.split]
        data_list = []
        for split in self.split:
            data_list += glob.glob(os.path.join(self.data_root, split, "*", "*"))
        return data_list

    @staticmethod
    def align_pose(coord, pose, target_pose):
        coord = np.hstack((coord, np.ones_like(coord[:, :1])))
        pose_align = np.matmul(np.linalg.inv(target_pose), pose)
        coord = (pose_align @ coord.T).T[:, :3]
        return coord

    def get_single_frame(self, idx):
        data_dict = super().get_data(idx)
        if self.with_images:
            data_dict["image_coord"] = repeat(
                data_dict["image_coord"],
                "n coord -> n cam coord",
                cam=data_dict["image_mask"].shape[1],
            )  # (N, CAM, 2)
            images = []
            for i in range(data_dict["image_mask"].shape[1]):
                data_path = self.data_list[idx % len(self.data_list)]
                image_path = os.path.join(data_path, f"image_{i}.jpg")
                images.append(imageio.imread(image_path))

                # some of the images are 1280x1920, some are 886x1920
                # we crop the top part of the 1280x1920 images
                H = images[-1].shape[0]
                assert H in (1280, 886)
                visible = data_dict["image_mask"][:, i]
                if H == 1280:
                    SKIP_IM_ROWS = 1280 - 886
                    images[-1] = images[-1][SKIP_IM_ROWS:]
                    data_dict["image_coord"][:, i, 1][visible] -= SKIP_IM_ROWS
                    visible &= data_dict["image_coord"][:, i, 1] >= 0
                data_dict["image_coord"][:, i][~visible] = -1
                data_dict["image_mask"][:, i] = visible

            data_dict["image"] = np.stack(images, axis=0)  # (CAM, H, W, 3)
        return data_dict

    def get_data(self, idx):
        idx = idx % len(self.data_list)
        if self.timestamp == (0,):
            return self.get_single_frame(idx)

        sequence_index = self.sequence_index[idx]
        lower, upper = self.sequence_offset[[sequence_index, sequence_index + 1]]
        major_frame = self.get_single_frame(idx)
        name = major_frame.pop("name")
        target_pose = major_frame.pop("pose")
        for key in major_frame.keys():
            major_frame[key] = [major_frame[key]]

        for timestamp in self.timestamp[1:]:
            refer_idx = timestamp + idx
            if refer_idx < lower or upper <= refer_idx:
                continue
            refer_frame = self.get_single_frame(refer_idx)
            refer_frame.pop("name")
            pose = refer_frame.pop("pose")
            refer_frame["coord"] = self.align_pose(
                refer_frame["coord"], pose, target_pose
            )
            if not self.reference_label:
                refer_frame["segment"] = (
                    np.ones_like(refer_frame["segment"]) * self.ignore_index
                )

            if self.timing_embedding:
                refer_frame["strength"] = np.hstack(
                    (
                        refer_frame["strength"],
                        np.ones_like(refer_frame["strength"]) * timestamp,
                    )
                )

            for key in major_frame.keys():
                major_frame[key].append(refer_frame[key])
        for key in major_frame.keys():
            major_frame[key] = np.concatenate(major_frame[key], axis=0)
        major_frame["name"] = name
        return major_frame

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        sequence_path, frame_name = os.path.split(file_path)
        sequence_name = os.path.basename(sequence_path)
        data_name = f"{sequence_name}_{frame_name}"
        return data_name
