import os
from omegaconf import OmegaConf
from hovsg.graph.graph import Graph
from hydra import initialize, compose
import open3d as o3d
import torch
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

import pickle
import open_clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from hovsg.utils.constants import MATTERPORT_GT_LABELS, CLIP_DIM

from hovsg.dataloader.hm3dsem import HM3DSemDataset
from hovsg.dataloader.scannet import ScannetDataset
from hovsg.dataloader.replica import ReplicaDataset
from hovsg.dataloader.truckscenes import TruckScenesDataset
import matplotlib.pyplot as plt
from hovsg.models.sam_clip_feats_extractor import extract_feats_per_pixel
import cv2

from hovsg.utils.graph_utils import (
    seq_merge,
    pcd_denoise_dbscan,
    feats_denoise_dbscan,
    distance_transform,
    map_grid_to_point_cloud,
    compute_room_embeddings,
    find_intersection_share,
    hierarchical_merge,
)
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

from hovsg.utils.clip_utils import get_img_feats, get_img_feats_batch
from hovsg.utils.sam_utils import crop_all_bounding_boxs, filter_masks
from hovsg.utils.graph_utils import (
    compute_3d_bbox_iou,
    find_overlapping_ratio_faiss,
    connected_components,
    merge_point_clouds_list
)
from matplotlib.patches import Rectangle

scenes_path = "/home/daniel/spatial_understanding/benchmarks/HOV-SG/data/splits/truckscenes_no_dark_no_highway_val.txt"
with open(scenes_path, 'r') as f:
    scenes = sorted([line.strip() for line in f.readlines()])
            
# scenes = [scenes[0]]
            
# Manually initialize Hydra and load the config
config_path = "../config"
config_name = "semantic_segmentation"
variation = "modified_seq_merge_params"

# Hydra context for manual loading
with initialize(version_base=None, config_path=config_path):
    params = compose(config_name=config_name)

# Create save directory
save_dir = os.path.join(params.main.save_path, params.main.dataset, variation)
os.makedirs(save_dir, exist_ok=True)

cfg = params


def filter_masks(masks, overlap_threshold):
    # 1) compute each mask’s area
    for mask in masks:
        seg = mask["segmentation"]
        mask["area"] = seg.sum()

    # 2) sort descending
    masks_sorted = sorted(masks, key=lambda m: m["area"], reverse=True)

    # 3) filter by >80% overlap
    overlap_threshold = 0.8
    kept_masks = []

    for m in masks_sorted:
        seg = m["segmentation"]
        area_m = m["area"]
        drop = False

        for km in kept_masks:
            # compute intersection
            inter = np.logical_and(seg, km["segmentation"]).sum()
            # if intersection covers >80% of the *current* (smaller) mask, drop it
            if inter / area_m > overlap_threshold:
                drop = True
                break

        if not drop:
            kept_masks.append(m)
            
    return kept_masks

# def extract_feats_per_pixel(
#     image,
#     mask_generator,
#     clip_model,
#     preprocess,
#     clip_feat_dim=768,
#     bbox_margin=0,
#     maskedd_weight=0.75,
# ):
#     """
#     Estimate the feature for each pixel in the image using ConceptFusion method.
#     """
#     LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH = image.shape[0], image.shape[1]
#     masks = mask_generator.generate(image)

#     # # ------------------------------------------------------------------------
#     # OVERLAP_THRESH = 0.8
#     # masks = filter_masks(masks, OVERLAP_THRESH)v


#     F_g = None
#     cropped_masked_feats = None
#     cropped_feats = None
#     if F_g is None and cropped_masked_feats is None and cropped_feats is None:
#         F_g = get_img_feats(image, preprocess, clip_model)
#         croped_images = crop_all_bounding_boxs(image, masks, block_background=False, bbox_margin=bbox_margin)
#         croped_images_masked = crop_all_bounding_boxs(image, masks, block_background=True, bbox_margin=bbox_margin)
#         number_of_masks = len(croped_images)

#         # for croped_image in croped_images:
#         #     plt.imshow(croped_image)
#         #     plt.axis("off")
#         #     plt.show()

#         cropped_masked_feats = get_img_feats_batch(croped_images_masked, preprocess, clip_model)
#         cropped_feats = get_img_feats_batch(croped_images, preprocess, clip_model)

#     fused_crop_feats = torch.from_numpy(
#         maskedd_weight * cropped_masked_feats + (1 - maskedd_weight) * cropped_feats
#     )
#     F_l = torch.nn.functional.normalize(fused_crop_feats, p=2, dim=-1).cpu().numpy()
#     if F_l.shape[0] == 0:
#         return None, None, None

#     cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
#     phi_l_G = cos(torch.from_numpy(F_l), torch.from_numpy(F_g))
#     w_i = torch.nn.functional.softmax(phi_l_G, dim=0).reshape(-1, 1)
#     F_p = w_i * F_g + (1 - w_i) * F_l.reshape(number_of_masks, clip_feat_dim)
#     F_p = torch.nn.functional.normalize(F_p, p=2, dim=-1)

#     F_p = F_p.cuda()
#     outfeat = torch.zeros(LOAD_IMG_HEIGHT * LOAD_IMG_WIDTH, clip_feat_dim, device="cuda")
#     non_zero_ids = torch.from_numpy(np.array([mask["segmentation"] for mask in masks])).reshape((len(masks), -1))
#     for i, mask in enumerate(masks):
#         non_zero_indices = torch.argwhere(non_zero_ids[i] == 1).cuda()
#         outfeat[non_zero_indices, :] += F_p[i, :]
#     outfeat = torch.nn.functional.normalize(outfeat, p=2, dim=-1)
#     outfeat = outfeat.half()
#     outfeat = outfeat.reshape((LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, clip_feat_dim))
#     return outfeat.cpu(), F_p.cpu(), masks, F_g


def extract_feats_per_pixel(
    image,
    mask_generator,
    clip_model,
    preprocess,
    clip_feat_dim=768,
    bbox_margin=0,
    maskedd_weight=0.75,
):
    """
    Estimate the feature for each pixel in the image using ConceptFusion method.
    Caches masks to disk for faster repeated runs.
    """
    LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH = image.shape[0], image.shape[1]
    masks = mask_generator.generate(image)

    F_g = None
    cropped_masked_feats = None
    cropped_feats = None
    if F_g is None and cropped_masked_feats is None and cropped_feats is None:
        F_g = get_img_feats(image, preprocess, clip_model)
        croped_images = crop_all_bounding_boxs(image, masks, block_background=False, bbox_margin=bbox_margin)
        croped_images_masked = crop_all_bounding_boxs(image, masks, block_background=True, bbox_margin=bbox_margin)
        number_of_masks = len(croped_images)

        cropped_masked_feats = get_img_feats_batch(croped_images_masked, preprocess, clip_model)
        cropped_feats = get_img_feats_batch(croped_images, preprocess, clip_model)

    fused_crop_feats = torch.from_numpy(
        maskedd_weight * cropped_masked_feats + (1 - maskedd_weight) * cropped_feats
    )
    F_l = torch.nn.functional.normalize(fused_crop_feats, p=2, dim=-1).cpu().numpy()
    if F_l.shape[0] == 0:
        return None, None, None

    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    phi_l_G = cos(torch.from_numpy(F_l), torch.from_numpy(F_g))
    w_i = torch.nn.functional.softmax(phi_l_G, dim=0).reshape(-1, 1)
    F_p = w_i * F_g + (1 - w_i) * F_l.reshape(number_of_masks, clip_feat_dim)
    F_p = torch.nn.functional.normalize(F_p, p=2, dim=-1)

    F_p = F_p.cuda()
    outfeat = torch.zeros(LOAD_IMG_HEIGHT * LOAD_IMG_WIDTH, clip_feat_dim, device="cuda")
    non_zero_ids = torch.from_numpy(np.array([mask["segmentation"] for mask in masks])).reshape((len(masks), -1))
    for i, mask in enumerate(masks):
        non_zero_indices = torch.argwhere(non_zero_ids[i] == 1).cuda()
        outfeat[non_zero_indices, :] += F_p[i, :]
    outfeat = torch.nn.functional.normalize(outfeat, p=2, dim=-1)
    outfeat = outfeat.half()
    outfeat = outfeat.reshape((LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, clip_feat_dim))
    return outfeat.cpu(), F_p.cpu(), masks, F_g

def process_scene(scene):
    global cfg
    
    # load the dataset
    dataset_cfg = {"root_dir": cfg.main.dataset_path, "transforms": None}
    if cfg.main.dataset == "hm3dsem":
        dataset = HM3DSemDataset(dataset_cfg)
    elif cfg.main.dataset == "scannet":
        dataset = ScannetDataset(dataset_cfg)
    elif cfg.main.dataset == "replica":
        dataset = ReplicaDataset(dataset_cfg)
    elif cfg.main.dataset == "truckscenes":
        dataset = TruckScenesDataset(dataset_cfg, scene)
    else:
        print("Dataset not supported")
        
    if dataset is None:
        print("No dataset loaded")
        
        
    tmp_folder = os.path.join(cfg.main.save_path, "tmp")
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    # for i in tqdm(range(0, len(dataset), cfg.pipeline.skip_frames), desc=f"Extracting features {scene}"):
    #     # Try to load precomputed features
    #     masks_path = os.path.join(tmp_folder, f"{scene}_masks_{i}.pkl")
    #     if os.path.exists(masks_path):
    #         try:
    #             with open(masks_path, "rb") as f:
    #                 F_2D, F_masks, masks, F_g = pickle.load(f)
    #             print(f"Loaded cached features from {masks_path}")
    #         except Exception as e:
    #             print(f"Error loading cached features from {masks_path}: {e}")
    #             exit(1)
    # return
                
                
    full_pcd = o3d.geometry.PointCloud()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load CLIP model
    if cfg.models.clip.type == "ViT-L/14@336px":
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained=str(cfg.models.clip.checkpoint),
            device=device,
        )
        clip_feat_dim = CLIP_DIM["ViT-L-14"]
        # clip_feat_dim = constants.clip_feat_dim[cfg.models.clip.type]
    elif cfg.models.clip.type == "ViT-H-14":
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14",
            pretrained=str(cfg.models.clip.checkpoint),
            device=device,
        )
        clip_feat_dim = CLIP_DIM["ViT-H-14"]
    clip_model.eval()
    if not hasattr(cfg, "pipeline"):
        print("-- entering querying and evaluation mode")
        # return


    # load the SAM model
    model_type = cfg.models.sam.type
    sam = sam_model_registry[model_type](
        checkpoint=str(cfg.models.sam.checkpoint)
    )
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=12,
        points_per_batch=144,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=50,
        crop_n_layers= 0,  # or 2 if needed
        # crop_n_points_downscale_factor= 4  # reduces resolution in crops
    )
    sam.eval()




    full_pcd = o3d.geometry.PointCloud()
    mask_feats = []
    mask_pcds = []
    full_feats_array = []


    # create the RGB-D point cloud
    for i in tqdm(range(0, len(dataset), cfg.pipeline.skip_frames), desc="Creating RGB-D point cloud"):
        rgb_image, depth_image, pose, _, depth_intrinsics = dataset[i]
        new_pcd = dataset.create_pcd(rgb_image, depth_image, pose)
        full_pcd += new_pcd

    # filter point cloud
    print("Size before filtering:", len(full_pcd.points))
    full_pcd = full_pcd.voxel_down_sample(
        voxel_size=cfg.pipeline.voxel_size
    )
    # full_pcd = pcd_denoise_dbscan(full_pcd, eps=0.01, min_points=100)
    print("Size after filtering:", len(full_pcd.points))
    # create tree from full point cloud
    locs_in = np.array(full_pcd.points)
    tree_pcd = cKDTree(locs_in)
    n_points = locs_in.shape[0]
    counter = torch.zeros((n_points, 1), device="cpu")
    sum_features = torch.zeros((n_points, clip_feat_dim), device="cpu")

    # extract features for each frame
    frame_save_path = os.path.join(save_dir, scene, "labelled_frames")
    os.makedirs(frame_save_path, exist_ok=True)
    
    frames_pcd = []
    frames_feats = []
    for i in tqdm(range(0, len(dataset), cfg.pipeline.skip_frames), desc=f"Extracting features {scene}"):
        rgb_image, depth_image, pose, _, _ = dataset[i]
        if rgb_image.size != depth_image.size:
            rgb_image = rgb_image.resize(depth_image.size)
            

        # Try to load precomputed features
        masks_path = os.path.join(tmp_folder, f"{scene}_masks_{i}.pkl")
        if os.path.exists(masks_path):
            with open(masks_path, "rb") as f:
                F_2D, F_masks, masks, F_g = pickle.load(f)
            print(f"Loaded cached features from {masks_path}")
        else:
            # Compute features
            F_2D, F_masks, masks, F_g = extract_feats_per_pixel(
                np.array(rgb_image),
                mask_generator,
                clip_model,
                preprocess,
                clip_feat_dim=clip_feat_dim,
                bbox_margin=cfg.pipeline.clip_bbox_margin,
                maskedd_weight=cfg.pipeline.clip_masked_weight,
            )
            # Save them
            # os.makedirs(os.path.dirname(masks_path), exist_ok=True)
            # with open(masks_path, "wb") as f:
            #     pickle.dump((F_2D, F_masks, masks, F_g), f)
            # print(f"Saved computed features to {masks_path}")

        F_2D = F_2D.cpu() 
           

        pcd = create_pcd(dataset, rgb_image, depth_image, pose)
        
        
        ###################
        masks_3d, masks_3d_feats = create_3d_masks(
            dataset,
            masks,
            F_masks,
            depth_image,
            full_pcd,
            tree_pcd,
            pose,
            down_size=cfg.pipeline.voxel_size,
            filter_distance=np.inf #cfg.pipeline.max_mask_distance,
        ) 
        
        # # Convert list of point clouds to list of numpy arrays
        # mask_path = os.path.join(frame_save_path, f"mask_points_{i}.pkl")
        # points_list = [np.asarray(pcd.points) for pcd in masks_3d]
        # if not os.path.exists(mask_path):
        #     with open(mask_path, "wb") as f:
        #         pickle.dump(points_list, f)
            
            
        # feat_path = os.path.join(frame_save_path, f"feat_{i}.npy")
        # if not os.path.exists(feat_path):
        #     np.save(feat_path, masks_3d_feats)
        #     ###############################
            
        # Prepare paths
        combined_path = os.path.join(frame_save_path, f"mask_and_feat_{i}.pkl")
        points_list = [np.asarray(scan.points) for scan in masks_3d]
        if not os.path.exists(combined_path):
            with open(combined_path, "wb") as f:
                pickle.dump({
                    "points": points_list,
                    "features": masks_3d_feats
                }, f)
            
        frames_pcd.append(masks_3d)
        frames_feats.append(masks_3d_feats)
        # fuse features for each point in the full pcd
        mask = np.array(depth_image) > 0
        mask = torch.from_numpy(mask)
        F_2D = F_2D[mask]
        # using cKdtree to find the closest point in the full pcd for each point in frame pcd
        dis, idx = tree_pcd.query(np.asarray(pcd.points), k=1, workers=-1)
        sum_features[idx] += F_2D
        counter[idx] += 1

        # Save individual masks and features for each frame
        # mask_path = os.path.join(frame_save_path, f"mask_{i}.ply")
        # if not os.path.exists(mask_path):
        #     o3d.io.write_point_cloud(mask_path, masks_3d)



    # compute the average features
    counter[counter == 0] = 1e-5
    sum_features = sum_features / counter
    full_feats_array = sum_features.cpu().numpy()
    full_feats_array: np.ndarray

    # del sum_features, counter
    # torch.cuda.empty_cache() 

    ########### MERGING & FUSION ###########

    # Merging the masks
    if cfg.pipeline.merge_type == "hierarchical":
        tqdm.write("Merging 3d masks hierarchically")
        mask_pcds = hierarchical_merge(
            frames_pcd, 
            cfg.pipeline.init_overlap_thresh, 
            cfg.pipeline.overlap_thresh_factor, 
            cfg.pipeline.voxel_size, 
            cfg.pipeline.iou_thresh,
        )
    elif cfg.pipeline.merge_type == "sequential":
        tqdm.write("Merging 3d masks sequentially") 
        mask_pcds = seq_merge(
            frames_pcd, 
            cfg.pipeline.init_overlap_thresh, 
            cfg.pipeline.voxel_size, 
            cfg.pipeline.iou_thresh
        )


    # remove any small pcds
    for i, pcd in enumerate(mask_pcds):
        if pcd.is_empty() or len(pcd.points) < cfg.pipeline.min_pcd_points:
            mask_pcds.pop(i)
            
    # fuse point features in every 3d mask
    masks_feats = []
    for i, mask_3d in tqdm(enumerate(mask_pcds), desc="Fusing features"):
        # find the points in the mask
        mask_3d = mask_3d.voxel_down_sample(cfg.pipeline.voxel_size * 2)
        points = np.asarray(mask_3d.points)
        dist, idx = tree_pcd.query(points, k=1, workers=-1)
        feats = full_feats_array[idx]
        feats = np.nan_to_num(feats)
        # filter feats with dbscan
        if feats.shape[0] == 0:
            masks_feats.append(
                np.zeros((1, clip_feat_dim), dtype=full_feats_array.dtype)
            )
            continue
        # print("Before", feats.shape)
        # feats = feats_denoise_dbscan(feats, eps=0.01, min_points=100)
        feats = feats_denoise_dbscan(feats, eps=0.01, min_points=5)
        # print("After", feats.shape)
        
        masks_feats.append(feats)
    mask_feats = masks_feats
    print("number of masks: ", len(mask_feats))
    print("number of pcds in hovsg: ", len(mask_pcds))
    assert len(mask_pcds) == len(mask_feats)


    state = "both"
    path = os.path.join(save_dir, scene)
    # def save_masked_pcds(self, path, state="both"):
    # """
    # Save the masked pcds to disk
    # :params state: str 'both' or 'objects' or 'full' to save the full masked pcds or only the objects.
    # """
    # # # remove any small pcds
    # tqdm.write("-- removing small and empty masks --")
    # for i, pcd in enumerate(mask_pcds):
    #     if len(pcd.points) < 50:
    #         mask_pcds.pop(i)
    #         mask_feats.pop(i)

    for i, pcd in enumerate(mask_pcds):
        if pcd.is_empty():
            mask_pcds.pop(i)
            mask_feats.pop(i)

    objects_path = os.path.join(path, "objects")


    # Save results
    if state == "both":
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(objects_path):
            os.makedirs(objects_path)
        print("number of masked pcds: ", len(mask_pcds))
        print("number of mask_feats: ", len(mask_feats))
        for i, pcd in enumerate(mask_pcds):
            o3d.io.write_point_cloud(
                os.path.join(objects_path, "pcd_{}.ply".format(i)), pcd
            )

        masked_pcd = o3d.geometry.PointCloud()
        for pcd in mask_pcds:
            pcd.paint_uniform_color(np.random.rand(3))
            masked_pcd += pcd
        o3d.io.write_point_cloud(os.path.join(path, "masked_pcd.ply"), masked_pcd)
        print("masked pcds saved to disk in {}".format(path))

    elif state == "objects":
        if not os.path.exists(path):
            os.makedirs(path)
        for i, pcd in enumerate(mask_pcds):
            o3d.io.write_point_cloud(
                os.path.join(objects_path, "pcd_{}.ply".format(i)), pcd
            )
        print("masked pcds saved to disk in {}".format(path))

    elif state == "full":
        if not os.path.exists(path):
            os.makedirs(path)
        masked_pcd = o3d.geometry.PointCloud()
        for pcd in mask_pcds:
            pcd.paint_uniform_color(np.random.rand(3))
            masked_pcd += pcd
        o3d.io.write_point_cloud(os.path.join(path, "masked_pcd.ply"), masked_pcd)
        print("masked pcds saved to disk in {}".format(path))

    # def save_full_pcd(self, path):
    # """
    # Save the full pcd to disk
    # :param path: str, The path to save the full pcd
    # """
    if not os.path.exists(path):
        os.makedirs(path)
    o3d.io.write_point_cloud(os.path.join(path, "full_pcd.ply"), full_pcd)
    print("full pcd saved to disk in {}".format(path))



    # def save_full_pcd_feats(self, path):
    # """
    # Save the full pcd with feats to disk
    # :param path: str, The path to save the full pcd feats
    # """
    if not os.path.exists(path):
        os.makedirs(path)
    # check if the full pcd feats is empty list
    if len(mask_feats) != 0:
        mask_feats = np.array(mask_feats)
        torch.save(
            torch.from_numpy(mask_feats), os.path.join(path, "mask_feats.pt")
        )
    if len(full_feats_array) != 0:
        torch.save(
            torch.from_numpy(full_feats_array),
            os.path.join(path, "full_feats.pt"),
        )
    print("full pcd feats saved to disk in {}".format(path))




def create_pcd(dataset, rgb, depth, camera_pose=None, mask_img=False, filter_distance=np.inf):
    """
    Create Open3D point cloud from RGB and depth images, and camera pose. filter_distance is used to filter out
    points that are further than a certain distance.
    :param rgb (pil image): RGB image
    :param depth (pil image): Depth image
    :param camera_pose (np.array): Camera pose
    :param mask_img (bool): Mask image
    :param filter_distance (float): Filter distance
    :return: Open3D point cloud
    """
    # convert rgb and depth images to numpy arrays
    rgb = np.array(rgb).astype(np.uint8)
    depth = np.array(depth)
    # resize rgb image to match depth image size if needed
    if rgb.shape[0] != depth.shape[0] or rgb.shape[1] != depth.shape[1]:
        rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
    # load depth camera intrinsics
    H = rgb.shape[0]
    W = rgb.shape[1]
    camera_matrix = dataset.depth_intrinsics
    scale = dataset.scale
    # create point cloud
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    depth = depth.astype(np.float32) / scale
    if mask_img:
        depth = depth * rgb
    mask = depth > 0
    x = x[mask]
    y = y[mask]
    depth = depth[mask]
    # convert to 3D
    X = (x - camera_matrix[0, 2]) * depth / camera_matrix[0, 0]
    Y = (y - camera_matrix[1, 2]) * depth / camera_matrix[1, 1]
    Z = depth
    if Z.mean() > filter_distance:
        return o3d.geometry.PointCloud()
    # convert to open3d point cloud
    points = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if not mask_img:
        colors = rgb[mask]
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    pcd.transform(camera_pose)
    return pcd

    
def create_3d_masks(dataset, masks, F_masks, depth, full_pcd, full_pcd_tree, camera_pose, down_size=0.02, filter_distance=None):
    """
    create 3d masks from 2D masks
    Args:
        masks: list of 2D masks
        depth: depth image
        full_pcd: full point cloud
        full_pcd_tree: KD-Tree of full point cloud
        camera_pose: camera pose
        down_size: voxel size for downsampling
    Returns:
        list of 3D masks as Open3D point clouds
    """
    pcd_list = []
    feat_list = []

    pcd = np.asarray(full_pcd.points)
    depth = np.array(depth)
    for i in range(len(masks)):
        # get the mask
        mask = masks[i]["segmentation"]
        mask = np.array(mask)
        # plt.imshow(mask, cmap='gray')
        # plt.imshow(depth, cmap='gray', alpha=0.5)
        # create pcd from mask
        pcd_masked = create_pcd(dataset, mask, depth, camera_pose, mask_img=True, filter_distance=filter_distance)

        
        pcd_masked = np.asarray(pcd_masked.points)
        dist, indices = full_pcd_tree.query(pcd_masked, k=1, workers=-1)
        pcd_masked = pcd[indices]
        pcd_mask = o3d.geometry.PointCloud()
        pcd_mask.points = o3d.utility.Vector3dVector(pcd_masked)
        colors = np.asarray(full_pcd.colors)
        colors = colors[indices]
        pcd_mask.colors = o3d.utility.Vector3dVector(colors)
        pcd_mask = pcd_mask.voxel_down_sample(voxel_size=down_size)
        
        if np.array(pcd_mask.points).shape[0] == 0:
            continue

        # using KD-Tree to find the nearest points in the point cloud
        # plt.figure(figsize=(10, 10))
        # plt.scatter(pcd[:, 0], pcd[:, 1], s=1, c='b', alpha=0.5)
        # plt.scatter(pcd_masked[:, 0], pcd_masked[:, 1], s=2, c='r', alpha=0.5)
        # plt.title(f"Masked Point Cloud {i}")
        # # plt.gca().set_aspect('equal')
        # plt.show()
        
        pcd_list.append(pcd_mask)
        feat_list.append(F_masks[i])
        
    return pcd_list, feat_list


def merge_3d_masks(mask_list, overlap_threshold=0.5, radius=0.02, iou_thresh=0.05):
    """
    merge the overlapped 3D masks in the list of masks using matrix
    :param pcd_list (list): list of point clouds
    :param overlap_threshold (float): threshold for overlapping ratio
    :param radius (float): radius for faiss search
    :param iou_thresh (float): threshold for iou
    :return: merged point clouds and features
    """
    
    aa_bb = [pcd.get_axis_aligned_bounding_box() for pcd in mask_list]
    overlap_matrix = np.zeros((len(mask_list), len(mask_list)))
    
    # fig, ax = plt.subplots()

    # create matrix of overlapping ratios
    for i in range(len(mask_list)):
        for j in range(i + 1, len(mask_list)):

            if compute_3d_bbox_iou(aa_bb[i], aa_bb[j]) > iou_thresh:
                overlap_matrix[i, j] = find_overlapping_ratio_faiss(mask_list[i], mask_list[j], radius=1.5 * radius)


                # plot_top_view_bboxes(ax, [aa_bb[i], aa_bb[j]])
                # print(np.array(mask_list[i].points).shape, np.array(mask_list[j].points).shape)
                # plt.scatter(np.array(mask_list[i].points)[:, 0], np.array(mask_list[i].points)[:, 1], s=1, c='blue', label='Mask ' + str(i))
                # plt.scatter(np.array(mask_list[j].points)[:, 0], np.array(mask_list[j].points)[:, 1], s=1, c='red', label='Mask ' + str(j))
                # plot_top_view_bboxes(ax, [aa_bb[j]], title="Top View Before Merging")
        #         break
        # break
    
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_aspect('equal', adjustable='box')
    # plt.grid(True)
    # plt.show()

    # check if overlap_matrix is zero size
    if overlap_matrix.size == 0:
        return mask_list
    graph = overlap_matrix > overlap_threshold
    n_components, component_labels = connected_components(graph)
    component_indices = [np.where(component_labels == i)[0] for i in range(n_components)]
    # merge the masks in each component
    pcd_list_merged = []
    for indices in component_indices:
        pcd_list_merged.append(merge_point_clouds_list([mask_list[i] for i in indices], voxel_size=0.5 * radius))

    # fig, ax = plt.subplots()
    # for merged_pcd in pcd_list_merged:
        # Plot the merged point cloud
        # ax.scatter(np.array(merged_pcd.points)[:, 0], np.array(merged_pcd.points)[:, 1], s=1, c=np.random.rand(3,), label='Merged Mask')
    # plt.title("Top View After Merging")
    # ax.set_aspect('equal', adjustable='box')
    # plt.show()

    return pcd_list_merged


def merge_adjacent_frames(frames_pcd, th, down_size, proxy_th):
    """
        Merge adjacent frames in the list of frames
        :param frames_pcd (list): list of point clouds
        :param th (float): threshold for overlapping ratio
        :param down_size (float): radius for downsampling
        :param proxy_th (float): threshold for iou
        :return: merged point clouds and features
    """
    new_frames_pcd = []
    for i in tqdm(range(0, len(frames_pcd), 2)):
        # if the number of frames is odd, the last frame is appended without merging.
        if i == len(frames_pcd) - 1:
            new_frames_pcd.append(frames_pcd[i])
            break
        pcd_list = frames_pcd[i] + frames_pcd[i + 1]

        pcd_list = merge_3d_masks(
            pcd_list,
            overlap_threshold=th,
            radius=down_size,
            iou_thresh=proxy_th,
        )
        new_frames_pcd.append(pcd_list)

    return new_frames_pcd

def hierarchical_merge(frames_pcd, th, th_factor, down_size, proxy_th):
    """
        Hierarchical merge the frames in the list of frames
        :param frames_pcd (list): list of point clouds
        :param th (float): threshold for overlapping ratio
        :param th_factor (float): factor for decreasing the threshold
        :param down_size (float): radius for downsampling
        :param proxy_th (float): threshold for iou
        :return: merged point clouds and features
    """
    while len(frames_pcd) > 1:
        frames_pcd = merge_adjacent_frames(frames_pcd, th, down_size, proxy_th)
        if len(frames_pcd) > 1:
            th -= th_factor * (len(frames_pcd) - 2) / max(1, len(frames_pcd) - 1)
            print("th: ", th)
        break
    # apply one more merge
    frames_pcd = frames_pcd[0]
    frames_pcd = merge_3d_masks(
        frames_pcd, overlap_threshold=0.75, radius=down_size, iou_thresh=proxy_th
    )
    return frames_pcd

import multiprocessing as mp
from tqdm import tqdm
import concurrent.futures

def wrapper(scene):
    tqdm.write(f"Processing scene: {scene}")
    process_scene(scene)
    tqdm.write(f"Scene {scene} processed successfully")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # <- Add this line!

    print("Starting semantic segmentation for TruckScenes dataset")
    print(f"Using config: {cfg}")
    print(f"Processing {len(scenes)} scenes")

    tqdm.write(f"Processing {len(scenes)} scenes")
    tqdm.write(f"Save directory: {save_dir}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(wrapper, scenes), total=len(scenes), desc="Processing scenes"))
