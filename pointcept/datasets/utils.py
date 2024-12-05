"""
Utils for Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from torchvision import tv_tensors


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        return default_collate(batch)


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    if "offset" in batch.keys():
        # Mix3d (https://arxiv.org/pdf/2110.02210.pdf)
        if random.random() < mix_prob:
            batch["unmix3d_offset"] = batch["offset"].clone()
            batch["offset"] = torch.cat(
                [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0
            )
    return batch


def gaussian_kernel(dist2: np.ndarray, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c**2))


def points2image(
    points: np.ndarray,  # (N, 3)
    points2cam: np.ndarray,  # (4, 4)
    K: np.ndarray,  # (3, 3)
    image_size: tuple[int, int],  # (H, W)
    depth: np.ndarray = None,  # (HxW,)
    min_distance: float = None,
    max_distance: float = None,
    error_margin: float = None,
) -> tuple[np.ndarray, np.ndarray]:  # (N, 2), (N,)
    # global to camera coords
    points_cam = points @ points2cam[:3, :3].T + points2cam[:3, 3]

    # points in front of camera
    visibility_mask = points_cam[:, 2] > 0

    # project
    points_image = points_cam @ K.T
    points_image[:, :2] = points_image[:, :2] / points_image[:, 2:]

    # within image size
    visibility_mask = np.logical_and(visibility_mask, np.all(points_image >= 0, axis=1))
    visibility_mask = np.logical_and(
        visibility_mask, points_image[:, 0] < image_size[1]
    )
    visibility_mask = np.logical_and(
        visibility_mask, points_image[:, 1] < image_size[0]
    )
    if min_distance is not None:
        visibility_mask = np.logical_and(
            visibility_mask, points_image[:, 2] > min_distance
        )
    if max_distance is not None:
        visibility_mask = np.logical_and(
            visibility_mask, points_image[:, 2] < max_distance
        )
    if depth is not None and error_margin is not None:
        points_depth = depth[
            points_image[visibility_mask][:, 1].astype(int),
            points_image[visibility_mask][:, 0].astype(int),
        ]

        visibility_mask[visibility_mask] = (
            np.abs(points_image[visibility_mask][:, 2] - points_depth) <= error_margin
        )

    return points_image[:, :2], visibility_mask


def tv_coord2bbox(image_coord, image):
    image_coord = torch.from_numpy(image_coord)
    return [
        tv_tensors.BoundingBoxes(
            torch.hstack(
                [image_coord[:, i], torch.ones_like(image_coord[:, i])]
            ).double(),  # type: ignore
            format="CXCYWH",
            canvas_size=image[i].shape[-2:],
        )
        for i in range(image_coord.shape[1])
    ]


def tv_bbox2coord(bbox):
    return torch.stack([b[:, :2] for b in bbox], dim=1).numpy()

def read_camera(camera_path):
    with open(camera_path, 'r') as file:
        data = file.read()
        cam_extr = np.fromstring(data, dtype=np.float32, sep=" ")

    z2y_top_m = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
    cam_t = np.matmul(z2y_top_m, cam_extr[:3] / 1000)
    if cam_extr.shape[0] > 3:
        cam_front, cam_up = cam_extr[3:6], cam_extr[6:9]
        cam_n = np.cross(cam_front, cam_up)
        cam_r = np.stack((cam_front, cam_up, cam_n), axis=1).astype(np.float32)
        cam_r = np.matmul(z2y_top_m, cam_r)
        cam_f = cam_extr[9:11]
    else:
        cam_r = np.eye(3, dtype=np.float32)
        cam_f = None
    return cam_r, cam_t, cam_f