"""
Structured3D Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import imageio.v3 as imageio
import numpy as np
from collections.abc import Sequence

from pointcept.datasets.utils import points2image, read_camera
from pointcept.utils.cache import shared_dict

from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class Structured3DDataset(DefaultDataset):
    def __init__(
        self,
        with_images=False,
        **kwargs,
    ):
        self.with_images = with_images
        super().__init__(**kwargs)

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(
                os.path.join(self.data_root, self.split, "scene_*/room_*")
            )
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(
                    os.path.join(self.data_root, split, "scene_*/room_*")
                )
        else:
            raise NotImplementedError
        return data_list

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, room_name = os.path.split(file_path)
        scene_name = os.path.basename(dir_path)
        data_name = f"{scene_name}_{room_name}"
        return data_name

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

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if self.with_images:
            H, W = (720, 1280)
            n_point = data_dict["coord"].shape[0]
            images = []
            image_coords = []
            image_masks = []
            scene, room = name.split("_room_")
            
            root_path = f"data/structured3d_images/{scene}/2D_rendering/{room}/perspective/full"
            frames = glob.glob(os.path.join(root_path, "*/"))
            if len(frames) < self.with_images:
                selected_images = [os.path.join(frame, "rgb_rawlight.png") for frame in frames]
                num_fill = self.with_images - len(selected_images)
            else:
                selected_indices = np.random.choice(
                    len(frames), self.with_images, replace=False
                )
                selected_images = [os.path.join(frames[i], "rgb_rawlight.png") for i in selected_indices]
                num_fill = 0
            for img_path in selected_images:
                image = imageio.imread(img_path)[:,:,:3]
                H, W = image.shape[:2]
                images.append(image)

                depth_path = os.path.join(
                    os.path.dirname(img_path),
                    os.path.basename(img_path).replace("rgb_rawlight", "depth"),
                )
                depth = imageio.imread(depth_path) / 1000

                extrinsic_path = os.path.join(
                    os.path.dirname(img_path),
                    os.path.basename(img_path).replace("rgb_rawlight.png", "camera_pose.txt"),
                )
                cam_r, cam_t, cam_f = read_camera(extrinsic_path)
                points2cam = np.eye(4)
                points2cam[:3, :3] = cam_r
                points2cam[:3, 3] = cam_t
                points2cam = np.linalg.inv(points2cam)
                a = np.eye(4)
                a[:3, :3] = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
                b = np.eye(4)
                b[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
                points2cam = a @ points2cam @ b

                intrinsic = np.eye(3)
                fx, fy = cam_f
                intrinsic[0, 2] = W / 2
                intrinsic[1, 2] = H / 2

                intrinsic[0, 0] = intrinsic[0, 2] / np.tan(fx)
                intrinsic[1, 1] = intrinsic[1, 2] / np.tan(fy)


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
            for _ in range(num_fill):
                images.append(np.zeros((H, W, 3), dtype=np.uint8))
                image_coords.append(np.zeros((n_point, 2), dtype=np.float32))
                image_masks.append(np.zeros((n_point), dtype=bool))
            data_dict["image"] = np.stack(images, axis=0)  # (CAM, H, W, 3)
            data_dict["image_coord"] = np.stack(image_coords, axis=1)  # (N, CAM, 2)
            data_dict["image_mask"] = np.stack(image_masks, axis=1)  # (N, CAM)


        return data_dict