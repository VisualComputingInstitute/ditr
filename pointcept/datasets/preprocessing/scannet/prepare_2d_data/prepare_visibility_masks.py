import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import numpy as np
from PIL import Image
from plyfile import PlyData
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# params
parser = argparse.ArgumentParser()
parser.add_argument("--images_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--downsampling_rate", type=int, default=10)
parser.add_argument("--image_size", type=list, default=[420, 560])
parser.add_argument("--min_distance", type=float, default=1.0)
parser.add_argument("--max_distance", type=float, default=4.0)
parser.add_argument("--error_margin", type=float, default=0.2)

opt = parser.parse_args()
print(opt)


def load_ply_coords(filepath):
    with open(filepath, "rb") as f:
        plydata = PlyData.read(f)
        data = plydata.elements[0].data
        coords = np.array([data["x"], data["y"], data["z"]], dtype=np.float32).T
    return coords


def points2image(
    points: np.ndarray,  # (N, 3)
    points2cam: np.ndarray,  # (4, 4)
    K: np.ndarray,  # (3, 3)
    image_size: tuple[int, int],  # (H, W)
    depth: np.ndarray,  # (HxW,)
    cut_corner_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:  # (N, 2), (N,)
    # global to camera coords
    points_cam = points @ points2cam[:3, :3].T + points2cam[:3, 3]

    # points in front of camera
    visibility_mask = points_cam[:, 2] > 0

    # project
    points_image = points_cam @ K.T
    points_image[:, :2] = points_image[:, :2] / points_image[:, 2:]

    # within image size
    visibility_mask = np.logical_and(
        visibility_mask, np.all(points_image[:, :2] >= cut_corner_threshold, axis=1)
    )
    visibility_mask = np.logical_and(
        visibility_mask, points_image[:, 0] < image_size[1] - cut_corner_threshold
    )
    visibility_mask = np.logical_and(
        visibility_mask, points_image[:, 1] < image_size[0] - cut_corner_threshold
    )
    visibility_mask = np.logical_and(
        visibility_mask, points_image[:, 2] < opt.max_distance
    )
    visibility_mask = np.logical_and(
        visibility_mask, points_image[:, 2] > opt.min_distance
    )

    depth = depth.reshape(image_size)
    points_depth = depth[
        points_image[visibility_mask][:, 1].astype(int),
        points_image[visibility_mask][:, 0].astype(int),
    ]

    visibility_mask[visibility_mask] = (
        np.abs(points_image[visibility_mask][:, 2] - points_depth) <= opt.error_margin
    )

    return points_image, visibility_mask


def process_scene(scene):
    try:
        ply_path = os.path.join(opt.images_path, scene, f"{scene}_vh_clean_2.ply")
        ply_coords = load_ply_coords(ply_path)
        depth_path = os.path.join(opt.images_path, scene, "depth")
        pose_path = os.path.join(opt.images_path, scene, "pose")
        intrinsic_path = os.path.join(
            opt.images_path, scene, "intrinsic/intrinsic_color.txt"
        )
        intrinsic = np.loadtxt(intrinsic_path)[:3, :3]
        scan = 0
        visibility_path = os.path.join(opt.output_path, scene, "visibility")
        os.makedirs(visibility_path, exist_ok=True)
        while os.path.exists(f"{depth_path}/{scan}.png"):
            depth_data = np.array(Image.open(f"{depth_path}/{scan}.png")) / 1000

            extrinsic = np.loadtxt(f"{pose_path}/{scan}.txt")

            h, w = depth_data.shape
            i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
            z = depth_data
            x = (i - intrinsic[0, 2]) * z / intrinsic[0, 0]
            y = (j - intrinsic[1, 2]) * z / intrinsic[1, 1]
            coords = np.stack((x, y, z), axis=-1).reshape(-1, 3)

            points2cam = np.linalg.inv(extrinsic)
            points_image, visibility_mask = points2image(
                ply_coords, points2cam, intrinsic, (h, w), coords[:, 2]
            )
            if visibility_mask.sum() > 0:
                np.save(os.path.join(visibility_path, f"{scan}_mask.npy"), visibility_mask)
                np.save(
                    os.path.join(visibility_path, f"{scan}_points.npy"), points_image[visibility_mask, :2]
                )
            scan += opt.downsampling_rate
        return f"{scene} processed successfully."

    except Exception as e:
        return f"Error processing {scene}: {e}"


if __name__ == "__main__":
    scenes = os.listdir(opt.images_path)
    with ProcessPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(process_scene, scene): scene for scene in scenes}
        for future in as_completed(futures):
            result = future.result()
            logging.info(result)
