import os
import struct
import numpy as np
import zlib
import imageio
import cv2
from concurrent.futures import ThreadPoolExecutor

COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {
    -1: "unknown",
    0: "raw_ushort",
    1: "zlib_ushort",
    2: "occi_ushort",
}


class RGBDFrame:
    def load(self, file_handle):
        self.camera_to_world = np.frombuffer(
            file_handle.read(16 * 4), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = file_handle.read(self.color_size_bytes)
        self.depth_data = file_handle.read(self.depth_size_bytes)

    def decompress_depth(self, compression_type):
        if compression_type == "zlib_ushort":
            return np.frombuffer(
                zlib.decompress(self.depth_data), dtype=np.uint16
            ).reshape(-1)
        else:
            raise ValueError("Unsupported compression type")

    def decompress_color(self, compression_type):
        if compression_type == "jpeg":
            return imageio.imread(self.color_data)
        else:
            raise ValueError("Unsupported compression type")


class SensorData:
    def __init__(self, filename):
        self.version = 4
        self.load(filename)

    def load(self, filename):
        with open(filename, "rb") as f:
            version = struct.unpack("I", f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack("Q", f.read(8))[0]
            self.sensor_name = f.read(strlen).decode("utf-8")
            self.intrinsic_color = np.frombuffer(
                f.read(16 * 4), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.frombuffer(
                f.read(16 * 4), dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.frombuffer(
                f.read(16 * 4), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.frombuffer(
                f.read(16 * 4), dtype=np.float32
            ).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[
                struct.unpack("i", f.read(4))[0]
            ]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[
                struct.unpack("i", f.read(4))[0]
            ]
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            num_frames = struct.unpack("Q", f.read(8))[0]
            self.frames = [RGBDFrame() for _ in range(num_frames)]
            for frame in self.frames:
                frame.load(f)

    def export_depth_images(self, output_path, image_size=None, frame_skip=1):
        os.makedirs(output_path, exist_ok=True)
        with ThreadPoolExecutor() as executor:
            for idx, frame in enumerate(self.frames[::frame_skip]):
                executor.submit(
                    self._save_depth_image,
                    frame,
                    output_path,
                    idx * frame_skip,
                    image_size,
                )

    def _save_depth_image(self, frame, output_path, index, image_size):
        depth = frame.decompress_depth(self.depth_compression_type).reshape(
            self.depth_height, self.depth_width
        )
        if image_size:
            depth = cv2.resize(
                depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST
            )
        imageio.imwrite(os.path.join(output_path, f"{index}.png"), depth)

    def export_color_images(self, output_path, image_size=None, frame_skip=1):
        os.makedirs(output_path, exist_ok=True)
        with ThreadPoolExecutor() as executor:
            for idx, frame in enumerate(self.frames[::frame_skip]):
                executor.submit(
                    self._save_color_image,
                    frame,
                    output_path,
                    idx * frame_skip,
                    image_size,
                )

    def _save_color_image(self, frame, output_path, index, image_size):
        color = frame.decompress_color(self.color_compression_type)
        if image_size:
            color = cv2.resize(
                color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST
            )
        imageio.imwrite(os.path.join(output_path, f"{index}.jpg"), color)

    def save_mat_to_file(self, matrix, filename):
        np.savetxt(filename, matrix, fmt="%f")

    def export_poses(self, output_path, frame_skip=1):
        os.makedirs(output_path, exist_ok=True)
        for idx, frame in enumerate(self.frames[::frame_skip]):
            self.save_mat_to_file(
                frame.camera_to_world,
                os.path.join(output_path, f"{idx*frame_skip}.txt"),
            )

    def export_intrinsics(self, output_path, image_size):
        os.makedirs(output_path, exist_ok=True)
        intrinsic_color, intrinsic_depth = self.adapt_intrinsic(image_size)
        self.save_mat_to_file(
            intrinsic_color, os.path.join(output_path, "intrinsic_color.txt")
        )
        self.save_mat_to_file(
            self.extrinsic_color, os.path.join(output_path, "extrinsic_color.txt")
        )
        self.save_mat_to_file(
            intrinsic_depth, os.path.join(output_path, "intrinsic_depth.txt")
        )
        self.save_mat_to_file(
            self.extrinsic_depth, os.path.join(output_path, "extrinsic_depth.txt")
        )

    def adapt_intrinsic(self, desired_resolution=[420, 560]):
        # Compute scale factors for width and height
        scale_x_color = desired_resolution[1] / self.color_width
        scale_y_color = desired_resolution[0] / self.color_height

        # Scale the focal lengths and the principal points
        intrinsic_color = self.intrinsic_color.copy()
        intrinsic_color[0, 0] *= scale_x_color  # fx
        intrinsic_color[1, 1] *= scale_y_color  # fy
        intrinsic_color[0, 2] *= scale_x_color  # cx
        intrinsic_color[1, 2] *= scale_y_color  # cy
        
        # Compute scale factors for width and height
        intrinsic_depth = self.intrinsic_depth.copy()
        scale_x_depth = desired_resolution[1] / self.depth_width
        scale_y_depth = desired_resolution[0] / self.depth_height

        # Scale the focal lengths and the principal points
        intrinsic_depth[0, 0] *= scale_x_depth  # fx
        intrinsic_depth[1, 1] *= scale_y_depth  # fy
        intrinsic_depth[0, 2] *= scale_x_depth  # cx
        intrinsic_depth[1, 2] *= scale_y_depth  # cy
        
        return intrinsic_color, intrinsic_depth