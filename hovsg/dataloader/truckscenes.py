import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from hovsg.dataloader.generic import RGBDDataset
import numpy as np
import open3d as o3d
import cv2

class TruckScenesDataset(RGBDDataset):
    """
    Dataset class for a LiDAR and multi-camera dataset.

    This class loads RGB images from multiple cameras and corresponding LiDAR point clouds,
    generates depth images by projecting points onto the image plane, and provides poses
    and intrinsics for each sample.
    """

    def __init__(self, cfg, scene):
        """
        Args:
            cfg: Configuration dictionary containing:
                - root_dir: Path to the root directory containing the dataset.
                - transforms: Optional transformations to apply to the data.
        """
        self.scene = scene
        super(TruckScenesDataset, self).__init__(cfg)
        self.root_dir = cfg["root_dir"]
        self.transforms = cfg["transforms"]
        # Depth scale set to 1.0 assuming LiDAR points are in meters
        self.scale = 1.0
        # Intrinsics will be loaded per sample, so initialize as None
        self.rgb_intrinsics = None
        self.depth_intrinsics = None
        
        self.data_list = self._get_data_list()

    def _get_data_list(self):
        """
        Generates a list of (scene, timestamp, camera) tuples representing all data samples.

        Returns:
            List of tuples (scene, timestamp, camera).
        """
        data_list = []
        trainval_dir = os.path.join(self.root_dir, "trainval")
        
        
        # for scene in os.listdir(trainval_dir):
        # for scene in scenes:
        scene_dir = os.path.join(trainval_dir, self.scene)
        if not os.path.isdir(scene_dir):
            raise ValueError(f"Scene directory {scene_dir} does not exist.")

        # Get timestamps from labelled_points directory
        labelled_points_dir = os.path.join(scene_dir, "labelled_points")
        timestamps = [f.split(".")[0] for f in os.listdir(labelled_points_dir) if f.endswith(".pth")]
        timestamps = sorted(timestamps, key=int)
        # Get camera names from images directory
        images_dir = os.path.join(scene_dir, "images")
        cameras = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
        for timestamp in timestamps:
            for camera in cameras:
                data_list.append((self.scene, timestamp, camera))
        return data_list

    def __getitem__(self, idx):
        """
        Loads a data sample for the given index.

        Args:
            idx: Index of the data sample.

        Returns:
            Tuple of (rgb_image, depth_image, pose, rgb_intrinsics, depth_intrinsics).
        """
        scene, timestamp, camera = self.data_list[idx]

        # Load RGB image
        img_path = os.path.join(self.root_dir, "trainval", scene, "images", camera, f"{timestamp}.png")
        rgb_image = self._load_image(img_path)

        # Load LiDAR point cloud
        pcd_path = os.path.join(self.root_dir, "trainval", scene, "labelled_points", f"{timestamp}.pth")
        points_xyz, _, sem_labels, instance_labels = torch.load(pcd_path, weights_only=False)

        # Load camera pose (assumed to be camera-to-world transformation)
        pose_path = os.path.join(self.root_dir, "trainval", scene, "poses", camera, f"{timestamp}.txt")
        pose = self._load_pose(pose_path)

        # Load camera intrinsics
        intr_path = os.path.join(self.root_dir, "trainval", scene, "intrinsics", camera, f"{timestamp}.txt")
        intrinsics = self._load_rgb_intrinsics(intr_path)
        # Set both RGB and depth intrinsics to the same camera intrinsics
        rgb_intrinsics = intrinsics
        depth_intrinsics = intrinsics
        # Generate depth image by projecting LiDAR points
        depth_image, points_xyz_filtered = self._create_depth_image(points_xyz, pose, intrinsics, rgb_image.size)
        self.depth_intrinsics = depth_intrinsics
        # pcd_from_depth_image = self.create_pcd(rgb_image, depth_image, pose)
        # Apply transformations if provided
        if self.transforms is not None:
            rgb_image = self.transforms(rgb_image)
            depth_image = self.transforms(depth_image)
            
        return rgb_image, depth_image, pose, rgb_intrinsics, depth_intrinsics
        # return pcd_from_depth_image, points_xyz_filtered

    def _load_image(self, path):
        """
        Loads an RGB image from the given path.

        Args:
            path: Path to the RGB image file.

        Returns:
            RGB image as a PIL Image.
        """
        return Image.open(path)

    def _load_depth(self, path):
        """
        Not implemented as depth is generated from LiDAR points.

        Args:
            path: Placeholder argument.

        Raises:
            NotImplementedError: Depth images are generated, not loaded.
        """
        raise NotImplementedError("Depth images are generated from LiDAR points.")

    def _load_pose(self, path):
        """
        Loads the camera pose from the given path.

        Args:
            path: Path to the pose file.

        Returns:
            Camera pose as a 4x4 NumPy array.
        """
        return np.loadtxt(path)

    def _load_rgb_intrinsics(self, path):
        """
        Loads the camera intrinsics from the given path.

        Args:
            path: Path to the intrinsics file.

        Returns:
            Intrinsics as a 3x3 NumPy array.
        """
        return np.loadtxt(path)

    def _load_depth_intrinsics(self, path):
        """
        Not implemented separately as depth intrinsics match RGB intrinsics.

        Args:
            path: Placeholder argument.

        Raises:
            NotImplementedError: Uses same intrinsics as RGB.
        """
        raise NotImplementedError("Depth intrinsics are the same as RGB intrinsics.")
    
    def _create_depth_image(self, points_world, camera_to_world, intrinsics, image_size):
        """
        Creates a depth image by projecting LiDAR points onto the image plane, based on compute_mapping logic.

        Args:
            points_world: LiDAR points in world coordinates (N, 3).
            camera_to_world: Camera-to-world pose (4x4 matrix).
            intrinsics: Camera intrinsics (3x3 matrix).
            image_size: Tuple of (width, height) of the RGB image.

        Returns:
            Depth image as a PIL Image.
        """
        width, height = image_size
        cut_bound = 0

        # Add homogeneous coordinates to points
        points_hom = np.hstack((points_world[:, :3], np.ones((points_world.shape[0], 1))))

        # Transform points to camera coordinates
        world_to_camera = np.linalg.inv(camera_to_world)
        points_cam = (world_to_camera @ points_hom.T).T

        # Extract coordinates
        x = points_cam[:, 0]
        y = points_cam[:, 1]
        z = points_cam[:, 2]

        # Filter points in front of the camera
        mask = z > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        points_world_filtered = points_world[mask]

        # Project to image plane
        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        u = (fx * x / z) + cx
        v = (fy * y / z) + cy

        # Round to integer pixel coordinates
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)

        # Filter points within image boundaries with cut_bound
        in_image = (u >= cut_bound) & (u < width - cut_bound) & (v >= cut_bound) & (v < height - cut_bound)
        u = u[in_image]
        v = v[in_image]
        z = z[in_image]
        points_world_filtered = points_world_filtered[in_image]

        # Initialize depth image with infinity
        depth_image = np.full((height, width), np.inf, dtype=np.float32)

        # print("v,u shape", v.shape)
        # Assign minimum depth per pixel efficiently
        np.minimum.at(depth_image, (v, u), z)

        # Replace infinity with 0 for no-data pixels
        depth_image[depth_image == np.inf] = 0


        # Count occupied pixels (depth > 0)
        # print(image_size)
        occupied_pixels = np.sum(depth_image > 0)
        # print("occupied_pixels", occupied_pixels)
        # print("points_world_filtered", points_world_filtered.shape)
    
        # Convert to PIL Image
        return Image.fromarray(depth_image), points_world_filtered
    
    

    def create__pcd(self, rgb, depth, camera_pose=None):
        """
        Creates a point cloud from RGB and depth images with camera pose for LiDARCameraDataset.

        Args:
            rgb: RGB image as a PIL Image.
            depth: Depth image as a PIL Image.
            camera_pose: Camera pose as a 4x4 NumPy array (optional).

        Returns:
            Open3D point cloud.
        """
        # Use self.depth_intrinsics; assumes it's set elsewhere if not passed per sample
        if not hasattr(self, 'depth_intrinsics') or self.depth_intrinsics is None:
            raise ValueError("Depth intrinsics must be set in LiDARCameraDataset instance")

        # Convert RGB and depth images to NumPy arrays
        rgb = np.array(rgb).astype(np.uint8)
        depth = np.array(depth).astype(np.float32)

        # Resize RGB image to match depth image size if necessary
        if rgb.shape[0] != depth.shape[0] or rgb.shape[1] != depth.shape[1]:
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)

        H, W = depth.shape
        camera_matrix = self.depth_intrinsics
        scale = self.scale  # Typically 1.0 for LiDARCameraDataset

        # Create meshgrid for pixel coordinates
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        depth = depth / scale

        # Mask for valid depth values
        mask = depth > 0
        x = x[mask]
        y = y[mask]
        depth = depth[mask]

        # Compute 3D points in camera coordinates
        X = (x - camera_matrix[0, 2]) * depth / camera_matrix[0, 0]
        Y = (y - camera_matrix[1, 2]) * depth / camera_matrix[1, 1]
        Z = depth

        # Create point cloud
        points = np.stack((X, Y, Z), axis=1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Assign colors from RGB image
        colors = rgb[mask]
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

        # Apply camera pose transformation if provided
        if camera_pose is not None:
            pcd.transform(camera_pose)

        return pcd