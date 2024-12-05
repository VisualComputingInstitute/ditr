import argparse
import os
import shutil
from SensorData import SensorData
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# params
parser = argparse.ArgumentParser()
parser.add_argument("--scannet_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--downsampling_rate", type=int, default=10)
parser.add_argument("--export_depth_images", type=bool, default=True)
parser.add_argument("--export_color_images", type=bool, default=True)
parser.add_argument("--export_poses", type=bool, default=True)
parser.add_argument("--export_intrinsics", type=bool, default=True)
parser.add_argument("--export_visibility", type=bool, default=True)
parser.add_argument("--image_size", type=list, default=[420, 560])
opt = parser.parse_args()
print(opt)


def process_scene(scene):
    try:
        sens_path = os.path.join(opt.scannet_path, scene, f"{scene}.sens")
        save_path = os.path.join(opt.output_path, scene)
        os.makedirs(save_path, exist_ok=True)

        ply_path = sens_path.replace(".sens", "_vh_clean_2.ply")
        shutil.copyfile(ply_path, os.path.join(save_path, f"{scene}_vh_clean_2.ply"))

        logger.info(f"loading {sens_path}...")
        sd = SensorData(sens_path)
        logger.info("loaded!")

        # Export data with configured settings
        if opt.export_depth_images:
            sd.export_depth_images(
                os.path.join(save_path, "depth"),
                image_size=opt.image_size,
                frame_skip=opt.downsampling_rate,
            )
        if opt.export_color_images:
            sd.export_color_images(
                os.path.join(save_path, "color"),
                image_size=opt.image_size,
                frame_skip=opt.downsampling_rate,
            )
        if opt.export_poses:
            sd.export_poses(
                os.path.join(save_path, "pose"), frame_skip=opt.downsampling_rate
            )
        if opt.export_intrinsics:
            sd.export_intrinsics(os.path.join(save_path, "intrinsic"), opt.image_size)

        return f"{scene} processed successfully."

    except Exception as e:
        return f"Error processing {scene}: {e}"


if __name__ == "__main__":
    scenes = os.listdir(opt.scannet_path)
    with ProcessPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(process_scene, scene): scene for scene in scenes}
        for future in as_completed(futures):
            result = future.result()
            logging.info(result)
