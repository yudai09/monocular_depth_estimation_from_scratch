from glob import glob
import gzip
import json
import os
from pathlib import Path
import pickle
from tqdm import tqdm

from torch.utils.data.dataset import Dataset
import cv2
import numpy as np
import pandas as pd
import transforms3d as t3d


class VKITTI2(Dataset):
    def __init__(
        self,
        root_dir="./data/",
        scenes=["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"],
        variations=["15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right",
                    "clone", "fog", "morning", "overcast", "rain", "sunset"],
        cameras=["Camera_0", "Camera_1"],
        frame_inds=[-1, 0]
    ):
        assert(type(cameras) == list)
        assert(type(scenes) == list)

        self.root_dir = Path(root_dir)
        self.scenes = scenes
        self.variations = variations
        self.cameras = cameras
        self.frame_inds = frame_inds
        print("loading sequences")
        self.load_sequences()
        self.build_samples()

    def load_sequences(self):
        self.sequences = []
        for scene in tqdm(self.scenes):
            for camera in self.cameras:
                for variation in self.variations:
                    variation_dir = self.root_dir / scene / variation
                    text_variation_dir = self.root_dir / "vkitti_2.0.3_textgt" / scene / variation

                    rgb_dir = variation_dir / "frames" / "rgb" / camera
                    depth_dir = variation_dir / "frames" / "depth" / camera
                    extrinsic_text = text_variation_dir / "extrinsic.txt"
                    intrinsic_text = text_variation_dir / "intrinsic.txt"

                    rgb_list = sorted(rgb_dir.glob("*.jpg"))
                    depth_list = sorted(depth_dir.glob("*.png"))

                    extrinsics = pd.read_csv(extrinsic_text, delimiter=' ')
                    intrinsics = pd.read_csv(intrinsic_text, delimiter=' ')
                    extrinsics = self.extrinsic_to_array(extrinsics[extrinsics["cameraID"] == int(camera[-1])])
                    intrinsics = self.intrinsic_to_array(intrinsics[intrinsics["cameraID"] == int(camera[-1])])
                    sequence = list(zip(rgb_list, depth_list, extrinsics, intrinsics))
                    self.sequences.append(sequence)

    def build_samples(self):
        self.samples = []
        for sequence in self.sequences:
            for i in range(len(sequence)):
                indices = [frame_idx + i for frame_idx in self.frame_inds]
                sample = []
                for idx in indices:
                    if 0 > idx or idx >= len(sequence):
                        break
                    sample.append(sequence[idx])
                else:
                    self.samples.append(sample)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        item = {}
        for i, idx in enumerate(self.frame_inds):
            item.update({
                f"rgb_{idx}": self.load_image(sample[i][0]),
                f"depth_{idx}": self.load_depth(sample[i][1]),
                f"extrinsic_{idx}": np.array(sample[i][2]),
                f"intrinsic_{idx}": np.array(sample[i][3])})
        return item

    def __len__(self):
        return len(self.samples)

    def load_image(self, path):
        return cv2.imread(str(path))

    def load_depth(self, path, scale=1/100):
        return cv2.imread(str(path), cv2.IMREAD_ANYDEPTH) * scale

    def extrinsic_to_array(self, extrinsics: pd.DataFrame):
        arry = []
        for _, row in extrinsics.iterrows():
            arry.append([
                [row["r1,1"], row["r1,2"], row["r1,3"], row["t1"]],
                [row["r2,1"], row["r2,2"], row["r2,3"], row["t2"]],
                [row["r3,1"], row["r3,2"], row["r3,3"], row["t3"]],
                [row["0"],    row["0.1"],  row["0.2"],  row["1"]]])
        return arry

    def intrinsic_to_array(self, intrinsics: pd.DataFrame):
        arry = []
        for _, row in intrinsics.iterrows():
            arry.append([
                [row["K[0,0]"],           0.0, row["K[0,2]"]],
                [0.0,           row["K[1,1]"], row["K[1,2]"]],
                [0.0,                     0.0,          1.0]])
        return arry


class Pandaset(Dataset):
    def __init__(
        self,
        root_dir="./data",
        variations=["camera", "lidar"],
        cameras=["back_camera", "front_camera", "front_left_camera", "front_right_camera", "left_camera", "right_camera"],
        frame_inds=[-1, 0]):

        self.root_dir = Path(root_dir)
        scenes = glob(f"{root_dir}/**")
        self.scenes = [os.path.split(scene)[-1] for scene in scenes]
        self.variations = variations
        self.cameras = cameras
        self.frame_inds = frame_inds
        print("loading sequences")
        self.load_sequences()
        self.build_samples()

    def load_sequences(self):
        self.sequences = []
        for scene in self.scenes:
            for camera in self.cameras:
                for variation in self.variations:
                    if variation == "camera":
                        rgb_dir = self.root_dir / scene / variation / camera 
                        pose_json_path = self.root_dir / scene / variation / camera / "poses.json"
                        intrinsic_json_path = self.root_dir / scene / variation / camera / "intrinsics.json"

                        rgb_list = sorted(rgb_dir.glob("*.jpg"))

                        extrinsics = self.pose_to_extrinsic_array(pose_json_path, len(rgb_list))
                        intrinsics = self.intrinsic_to_array(intrinsic_json_path, len(rgb_list))
                        sequence = list(zip(rgb_list, extrinsics, intrinsics))
                        self.sequences.append(sequence)
                    elif variation == "lidar":
                        pass

    def build_samples(self):
        self.samples = []
        for sequence in self.sequences:
            for i in range(len(sequence)):
                indices = [frame_idx + i for frame_idx in self.frame_inds]
                sample = []
                for idx in indices:
                    if 0 > idx or idx >= len(sequence):
                        break
                    sample.append(sequence[idx])
                else:
                    self.samples.append(sample)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        item = {}
        for i, idx in enumerate(self.frame_inds):
            item.update({
                f"rgb_{idx}": self.load_image(sample[i][0]),
                f"extrinsic_{idx}": np.array(sample[i][1]),
                f"intrinsic_{idx}": np.array(sample[i][2])})
        return item

    def __len__(self):
        return len(self.samples)

    def load_image(self, path):
        return cv2.imread(str(path))
    
    def load_json(self, json_path):
        with open(json_path, "r") as f:
            out = json.load(f)
        return out
    
    def pose_to_extrinsic_array(self, pose_json_path, len_rgb_list):
        arr = []
        for idx in range(len_rgb_list):
            pose = self.load_json(pose_json_path)[idx]
            # クォータニオン
            quat = np.array([pose["heading"]["w"],
                             pose["heading"]["x"],
                             pose["heading"]["y"],
                             pose["heading"]["z"]])
            # カメラ位置（ワールド座標）
            world_pos = np.array([pose["position"]["x"],
                                  pose["position"]["y"],
                                  pose["position"]["z"]])
            # Compose translations(移動), rotations, zooms, [shears] to affine
            pose_mat = t3d.affines.compose(np.array(world_pos),
                                           t3d.quaternions.quat2mat(quat), # Calculate rotation matrix corresponding to quaternion
                                           [1.0, 1.0, 1.0])
            arr.append(np.linalg.inv(pose_mat).tolist())
        return arr
        
    def intrinsic_to_array(self, intrinsic_json_path, len_rgb_list):
        arr = []
        intrinsic = self.load_json(intrinsic_json_path)
        intrinsic_array = np.array([
            [intrinsic["fx"], 0, intrinsic["cx"]],
            [0, intrinsic["fy"], intrinsic["cy"]],
            [0, 0, 1]
        ])
        arr.append(intrinsic_array.tolist())
        return arr * len_rgb_list