import argparse
import json
import shutil
import numpy as np
import os
import imageio.v3 as imageio
from pathlib import Path
from pointcept.datasets.utils import points2image
from concurrent.futures import ProcessPoolExecutor, as_completed


S3DIS_PATH = Path("data/s3dis")
AREA_MAPPING = {
    "Area_1": ("area_1",),
    "Area_2": ("area_2",),
    "Area_3": ("area_3",),
    "Area_4": ("area_4",),
    "Area_5": ("area_5a", "area_5b"),
    "Area_6": ("area_6",),
}

AREA_5B_CORRECTION = np.array(
    [
        [0, -1, 0, 6.22617759],
        [1, 0, 0, 4.09703582],
        [0, 0, 1, 0.0],
        [0, 0, 0, 1.0],
    ]
)


def copy_file(src: Path, dst: Path):
    if dst.exists():
        print(f"Skipping existing file: {dst}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    return True

def process_room(raw_root, output_area, room, angle, output_path):
    print(f"Processing {output_area} - {room} with angle {angle}")

    coords = np.load(S3DIS_PATH / output_area / room / "coord.npy")
    room_center = (np.max(coords, axis=0) + np.min(coords, axis=0)) / 2
    assert angle in [0, 90, 180, 270]
    angle = (2 - angle / 180) * np.pi
    rot_cos, rot_sin = np.cos(angle), np.sin(angle)
    rot_t = np.array(
        [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]]
    )

    images = []
    for area in AREA_MAPPING[output_area]:
        rgb_dir = raw_root / area / "data" / "rgb"

        assert rgb_dir.exists(), f"RGB directory does not exist: {rgb_dir}"
        images += sorted(rgb_dir.glob(f"*_{room}_frame_*_domain_rgb.png"))

    frame_idx = 0
    for rgb_path in images:
        data_root = rgb_path.parents[1]
        filename = rgb_path.name
        depth_path = data_root / "depth" / filename.replace(
            "_domain_rgb.png", "_domain_depth.png"
        )
        pose_path = data_root / "pose" / filename.replace(
            "_domain_rgb.png", "_domain_pose.json"
        )
        depth = imageio.imread(depth_path) / 512.0
        H, W = depth.shape[:2]

        with open(pose_path, "r") as f:
            pose_data = json.load(f)

        extrinsic = np.eye(4)
        extrinsic[:3, :4] = np.asarray(pose_data["camera_rt_matrix"])
        if rgb_path.parents[2].name == "area_5b":
            extrinsic = extrinsic @ AREA_5B_CORRECTION

        extrinsic = np.linalg.inv(extrinsic)
        extrinsic[:3, :3] = rot_t @ extrinsic[:3, :3]
        extrinsic[:3, 3] = (extrinsic[:3, 3] - room_center) @ np.transpose(
            rot_t
        ) + room_center
        extrinsic = np.linalg.inv(extrinsic)

        intrinsic = np.array(pose_data["camera_k_matrix"])

        points_image, visibility_mask = points2image(
            coords,
            extrinsic,
            intrinsic,
            (H, W),
            depth=depth,
            min_distance=None,
            max_distance=None,
            error_margin=0.2,
        )

        if np.any(visibility_mask):
            room_output_path = output_path / output_area / room
            rgb_save_path = room_output_path / "color" / f"{frame_idx}.png"
            mask_save_path = (
                room_output_path / "visibility" / f"{frame_idx}_mask.npy"
            )
            points_save_path = (
                room_output_path / "visibility" / f"{frame_idx}_points.npy"
            )

            (room_output_path / "visibility").mkdir(parents=True, exist_ok=True)
            copy_file(rgb_path, rgb_save_path)
            visible_idx = np.flatnonzero(visibility_mask).astype(np.int32)
            np.save(mask_save_path, visible_idx)
            np.save(points_save_path, points_image[visible_idx, :2].astype(np.float32))
            frame_idx += 1
    return output_area, room, frame_idx, len(images)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s3dis_root",
        type=Path,
        required=True,
        help="Path to Stanford3dDataset_v1.2 dataset",
    )
    parser.add_argument(
        "--raw_root",
        type=Path,
        required=True,
        help="Path to Stanford2d3dDataset_noXYZ dataset",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/s3dis_images"),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers for processing rooms",
    )
    args = parser.parse_args()

    tasks = []

    for output_area in AREA_MAPPING.keys():
        area_info = np.loadtxt(
            os.path.join(
                args.s3dis_root,
                output_area,
                f"{output_area}_alignmentAngle.txt",
            ),
            dtype=str,
        )
        room_angles = {room_info[0]: int(room_info[1]) for room_info in area_info}

        for room, angle in room_angles.items():
            tasks.append(
                (args.raw_root, output_area, room, angle, args.output_path)
            )

    print(f"Found {len(tasks)} rooms")
    print(f"Using {args.num_workers} workers")

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(process_room, *task)
            for task in tasks
        ]

        for future in as_completed(futures):
            output_area, room, saved_count, total_images = future.result()
            print(
                f"Done {output_area} - {room}: "
                f"{saved_count}/{total_images} images saved"
            )

if __name__ == "__main__":
    main()
