
import json
import os
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import open3d as o3d
import open_clip
import plyfile
from scipy.spatial import cKDTree
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import BallTree
import torch
import torchmetrics as tm
from hydra import initialize, compose

from hovsg.labels.label_constants import (
    SCANNET_COLOR_MAP_20, 
    SCANNET_LABELS_20, 
    TRUCKSCENES_LABELS,
    TRUCKSCENES_COLORMAP
)

from hovsg.utils.eval_utils import (
    load_feature_map,
    knn_interpolation,
    read_ply_and_assign_colors,
    read_semantic_classes,
    sim_2_label,
    read_semantic_classes_replica,
    text_prompt,
    read_ply_and_assign_colors_replica
)
from hovsg.utils.metric import (
    frequency_weighted_iou,
    mean_iou,
    mean_accuracy,
    pixel_accuracy,
    per_class_iou,
)
from scipy.spatial import cKDTree

IGNORE_CLASS_INDEX = 12
TRUCKSCENES_LABELS_TO_IDX = {
    "animal": 0,
    "human.pedestrian.adult": 7,
    "human.pedestrian.child": 7,
    "human.pedestrian.construction_worker": 7,
    "human.pedestrian.personal_mobility": 7,
    "human.pedestrian.police_officer": 7,
    "human.pedestrian.stroller": IGNORE_CLASS_INDEX,
    "human.pedestrian.wheelchair": IGNORE_CLASS_INDEX,
    "movable_object.barrier": 1,
    "movable_object.debris": IGNORE_CLASS_INDEX,
    "movable_object.pushable_pullable": IGNORE_CLASS_INDEX,
    "movable_object.trafficcone": 8,
    "static_object.bicycle_rack": IGNORE_CLASS_INDEX,
    "static_object.traffic_sign": 9,
    "vehicle.bicycle": 2,
    "vehicle.bus.bendy": 3,
    "vehicle.bus.rigid": 3,
    "vehicle.car": 4,
    "vehicle.construction": 6,
    "vehicle.emergency.ambulance": IGNORE_CLASS_INDEX,
    "vehicle.emergency.police": IGNORE_CLASS_INDEX,
    "vehicle.motorcycle": 5,
    "vehicle.trailer": 10,
    "vehicle.truck": 11,
    "vehicle.train": IGNORE_CLASS_INDEX,
    "vehicle.other": IGNORE_CLASS_INDEX,
    "vehicle.ego_trailer": IGNORE_CLASS_INDEX,
    "unlabeled": IGNORE_CLASS_INDEX
}

TRUCKSCENES_LABELS = (
    'animal', 
    'barrier', 
    'bicycle', 
    'bus', 
    'car', 
    'motorcycle', 
    'construction vehicle', 
    'person', 
    'traffic cone',
    'traffic sign', 
    'trailer', 
    'truck', 
    'other'
)
    
# Manually initialize Hydra and load the config
config_path = "../../config"
config_name = "eval_sem_seg"

# Hydra context for manual loading
with initialize(version_base=None, config_path=config_path):
    params = compose(config_name=config_name)


scenes_path = "/home/daniel/spatial_understanding/benchmarks/HOV-SG/data/splits/truckscenes_no_dark_no_highway_val.txt"
with open(scenes_path, 'r') as f:
    scenes = sorted([line.strip() for line in f.readlines()])

# scenes = scenes[:5]      
# scenes = [scenes[0]]

def process_scene(scene_name):
    global params
    
    # load CLIP model
    if params.models.clip.type == "ViT-L/14@336px":
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained=str(params.models.clip.checkpoint),
            device=params.main.device,
        )
        clip_feat_dim = 768
    elif params.models.clip.type == "ViT-H-14":
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14",
            pretrained=str(params.models.clip.checkpoint),
            device=params.main.device,
        )
        clip_feat_dim = 1024
    clip_model.eval()

    # Load Feature Map
    feature_map_path = os.path.join(params.main.feature_map_path, scene_name)
    masked_pcd, mask_feats = load_feature_map(feature_map_path)


    TRUCKSCENES_LABELS_list = list(TRUCKSCENES_LABELS)
    labels_id = list(TRUCKSCENES_COLORMAP.keys())

    sim = text_prompt(clip_model, clip_feat_dim, mask_feats, TRUCKSCENES_LABELS_list, templates=True)
    labels = sim_2_label(sim, labels_id)
    labels = np.array(labels)
    
        
    # create a new pcd from the labeld pcd masks
    colors = np.array([TRUCKSCENES_COLORMAP[i] for i in labels]) / 255.0
    colors_map = TRUCKSCENES_COLORMAP
    colors_map = {int(k): np.array(v) / 255.0 for k, v in colors_map.items()}

    # load ground truth pcd
    scene_map_path = f"/shared/data/truckScenes/truckscenes_converted/trainval/{scene_name}/labelled_map.pth"
    xyz, feats, label, inst_label = torch.load(scene_map_path, weights_only=False)
    label = np.array([TRUCKSCENES_LABELS_TO_IDX.get(l, IGNORE_CLASS_INDEX) for l in label], dtype=np.int64)

    # Get corresponding labels for the masks using kdtree
    full_pcd = xyz
    full_pcd_kdtree = cKDTree(full_pcd[:, :3])

    # Create a mask for the labels that are not within the masked pcd
    mask = np.zeros(len(full_pcd), dtype=bool)

    masked_pcd_modified = []
    gt_labels = []
    gt_points = []

    for pcd_mask in masked_pcd:
        # Query neighbors for all masked_pcd points at once
        dists, idxs = full_pcd_kdtree.query(np.array(pcd_mask.points), k=1)
        
        # Create a new point cloud with the masked points
        pcd_mask_o3d = o3d.geometry.PointCloud()
        pcd_mask_o3d.points = o3d.utility.Vector3dVector(xyz[idxs, :3])
        masked_pcd_modified.append(pcd_mask_o3d)
        
        gt_labels.append(label[idxs])
        gt_points.append(xyz[idxs, :3])
        
    gt_labels = np.concatenate(gt_labels, axis=0)
    gt_points = np.concatenate(gt_points, axis=0)


    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points)

    # assing colors to gt pcd based on labels
    colors = np.zeros((len(gt_labels), 3))
    for i, label in enumerate(gt_labels):
        colors[i] = colors_map[label]
    gt_pcd.colors = o3d.utility.Vector3dVector(colors)


    ## FOR MASK BASED SEGMENTATION ##
    pcd = o3d.geometry.PointCloud()
    for i in range(len(masked_pcd)):
        pcd += masked_pcd_modified[i].paint_uniform_color(colors[i])

    pred_labels = []
    for i in range(len(masked_pcd_modified)):
        pred_labels.append(np.repeat(labels[i], len(masked_pcd_modified[i].points)))

    pred_labels = np.hstack(pred_labels)
    pred_labels = pred_labels.reshape(-1, 1)

    gt_labels = gt_labels.reshape(-1, 1)


    # concat coords and labels for predicied pcd
    coords_labels = np.zeros((len(pcd.points), 4))
    coords_labels[:, :3] = np.asarray(pcd.points)
    coords_labels[:, -1] = pred_labels[:, 0]
    # concat coords and labels for gt pcd
    coords_gt = np.zeros((len(gt_pcd.points), 4))
    coords_gt[:, :3] = np.asarray(gt_pcd.points)
    coords_gt[:, -1] = gt_labels[:, 0]
    # knn interpolation
    match_pc = knn_interpolation(coords_labels, coords_gt, k=5)
    pred_labels = match_pc[:, -1].reshape(-1, 1)
    ## MATCHING ##
    labels_gt = gt_labels
    label_pred = pred_labels
    assert len(labels_gt) == len(pred_labels)
    labels_gt, label_pred = gt_labels, pred_labels


    # print("labels_gt", labels_gt, "pred_labels", pred_labels)
    ignore = [IGNORE_CLASS_INDEX]

    # print(label_pred, labels_gt, ignore)
    print("################ {} ################".format(scene_name))
    ious = per_class_iou(label_pred, labels_gt, ignore=ignore, classes=TRUCKSCENES_LABELS)
    # print("per class iou: ", ious)
    miou = mean_iou(label_pred, labels_gt, ignore=ignore)
    print("miou: ", miou)
    fmiou = frequency_weighted_iou(label_pred, labels_gt, ignore=ignore)
    print("fmiou: ", fmiou)
    macc = mean_accuracy(label_pred, labels_gt, ignore=ignore)
    print("macc: ", macc)
    pacc = pixel_accuracy(label_pred, labels_gt, ignore=ignore)
    print("pacc: ", pacc)
    
    return {
        "scene_name": scene_name,
        "ious": ious,
        "miou": miou,
        "fmiou": fmiou,
        "macc": macc,
        "pacc": pacc,
        "pred_labels": label_pred.squeeze(), 
        "gt_labels":  labels_gt.squeeze(),   
    }


import multiprocessing as mp
from tqdm import tqdm
import concurrent.futures


def wrapper(scene):
    tqdm.write(f"Processing scene: {scene}")
    result = process_scene(scene)
    tqdm.write(f"Scene {scene} processed successfully")
    return result # Return the result from process_scene

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    print("Starting semantic segmentation for TruckScenes dataset")
    print(f"Using config: {params}")
    print(f"Processing {len(scenes)} scenes")

    tqdm.write(f"Processing {len(scenes)} scenes")

    all_scene_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        # executor.map returns an iterator, so convert to list to get all results
        # tqdm will wrap this iterator to show progress
        all_scene_results = list(tqdm(executor.map(wrapper, scenes), total=len(scenes), desc="Processing scenes"))

        
        
        
    # Now, calculate the mean metrics from all_scene_results
    if all_scene_results:
        # Compute mean metrics for single-value metrics
        mean_miou = np.mean([result["miou"] for result in all_scene_results])
        mean_fmiou = np.mean([result["fmiou"] for result in all_scene_results])
        mean_macc = np.mean([result["macc"] for result in all_scene_results])
        mean_pacc = np.mean([result["pacc"] for result in all_scene_results])

        # Collect all per-class IoUs into a list
        all_ious = [result["ious"] for result in all_scene_results]
        # Convert to a 2D NumPy array (num_scenes x num_classes)
        all_ious_array = np.array(all_ious)
        
        # Compute mean per-class IoUs, ignoring NaNs
        mean_class_ious = np.nanmean(all_ious_array, axis=0)

        # Print results
        print("\n--- Overall Mean Metrics ---")
        print(f"Mean mIoU: {mean_miou:.4f}")
        print(f"Mean fIoU: {mean_fmiou:.4f}")
        print(f"Mean mAcc: {mean_macc:.4f}")
        print(f"Mean pAcc: {mean_pacc:.4f}")
        print("Mean IoUs per class:")
        for i, mean_iou_val in enumerate(mean_class_ious):
            print(f"  Class {TRUCKSCENES_LABELS[i]}: {mean_iou_val:.4f}")
            
        all_preds = []
        all_gts   = []
        for res in all_scene_results:
            all_preds.append(res["pred_labels"])
            all_gts.append(  res["gt_labels"])
        
        # Concatenate into single big arrays:
        all_preds = np.concatenate(all_preds, axis=0)
        all_gts   = np.concatenate(all_gts,   axis=0)
        
        # Save the results for evaluation with other methods
        results_save_path = f"/home/daniel/spatial_understanding/benchmarks/inference/hov-sg/val_per_scene.npz"

        # Save everything into one compressed .npz:
        np.savez_compressed(
            results_save_path,
            preds=all_preds,
            gts=all_gts,
        )
        print("Wrote all preds & gts to truckscenes_eval_all_scenes.npz")
    
    
    
    else:
        print("No scene results to process.")
    
    # if all_scene_results:
    #     # Initialize sums for each metric
    #     total_miou = 0.0
    #     total_fmiou = 0.0
    #     total_macc = 0.0
    #     total_pacc = 0.0
        
    #     # For per-class IoU, we need to sum up elements at corresponding indices.
    #     # Initialize with zeros, assuming the number of classes is consistent.
    #     # Get the number of classes from the first scene's result.
    #     num_classes = len(all_scene_results[0]["ious"])
    #     class_ious_sums = np.zeros(num_classes)
        
    #     # Iterate through the results to sum up values
    #     for result in all_scene_results:

    #         total_miou += result["miou"]
    #         total_fmiou += result["fmiou"]
    #         total_macc += result["macc"]
    #         total_pacc += result["pacc"]
            
    #         # Add the per-class IoU list directly to the sum array
    #         class_ious_sums += np.array(result["ious"])

    #     num_scenes = len(all_scene_results)

    #     # Calculate means
    #     mean_miou = total_miou / num_scenes
    #     mean_fmiou = total_fmiou / num_scenes
    #     mean_macc = total_macc / num_scenes
    #     mean_pacc = total_pacc / num_scenes
        
    #     # Mean per-class IoUs
    #     mean_class_ious = class_ious_sums / num_scenes

    #     print("\n--- Overall Mean Metrics ---")
    #     print(f"Mean mIoU: {mean_miou:.4f}")
    #     print(f"Mean fIoU: {mean_fmiou:.4f}")
    #     print(f"Mean mAcc: {mean_macc:.4f}")
    #     print(f"Mean pAcc: {mean_pacc:.4f}")
    #     print("Mean IoUs per class:")
    #     # Assuming your classes are implicitly ordered by their index
    #     # If you have actual class names, you'd need to pass them around.
    #     for i, mean_iou_val in enumerate(mean_class_ious):
    #         print(f"  Class {TRUCKSCENES_LABELS[i]}: {mean_iou_val:.4f}")
    # else:
    #     print("No scene results to process.")
