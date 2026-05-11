"""
Convert DPRecon multi-view data to ShapeR pkl format.

DPRecon directory layout expected:
  cameras.npz            -- world_mat_i, scale_mat_i for each frame
  traj.txt               -- one 4x4 c2w pose per line (2000 lines for 10 frames)
  000{i}_rgb.png         -- RGB image, 384x384
  000{i}_depth.npy       -- normalized depth, range [0,1]; depth_m = value * scale_s (ray-length)
  instance_mask/000{i}.png -- instance segmentation mask, pixel value = instance id
  instance_id.json       -- {object_name: instance_id}
  replica_scan1_prompt_geometry.json -- {str_idx: description}

Usage:
  python preprocessing/dprecon_to_pkl.py --input_dir ~/Downloads/replica-scan1 --output_dir data/
  python preprocessing/dprecon_to_pkl.py --input_dir ~/Downloads/replica-scan1 --depth_dir ~/Downloads/depths --output_dir data/
"""

import argparse
import glob
import io
import json
import os
import pickle

import numpy as np
import torch
from PIL import Image


# Traj line indices corresponding to each of the 10 frames (0-9)
TRAJ_LINE_INDICES = [0, 120, 240, 600, 720, 960, 1080, 1200, 1560, 1680]

# Replica original image size (1200x680) -> center-crop 680x680 -> resize 384x384
ORIG_W, ORIG_H = 1200, 680
CROP_SIZE = 680  # center-crop to square
OUT_SIZE = 384
_scale = OUT_SIZE / CROP_SIZE
_x_offset = (ORIG_W - CROP_SIZE) / 2
_fx_orig = 600.0
_cx_orig = (ORIG_W - 1) / 2  # 599.5
_cy_orig = (ORIG_H - 1) / 2  # 339.5

K_384 = np.array([
    [_fx_orig * _scale,         0.,  (_cx_orig - _x_offset) * _scale],
    [        0.,         _fx_orig * _scale,          _cy_orig * _scale],
    [        0.,                 0.,                          1.      ],
], dtype=np.float32)

MAX_POINTS = 2048   # target point cloud size per object
MIN_VISIBLE_PX = 50  # skip frame if fewer than this many object pixels


def load_poses(traj_path):
    lines = open(traj_path).readlines()
    poses = []
    for idx in TRAJ_LINE_INDICES:
        row = np.array(list(map(float, lines[idx].split())), dtype=np.float64)
        poses.append(row.reshape(4, 4))
    return poses  # list of 10 T_c2w matrices


def resolve_depth_path(depth_dir, frame_idx):
    """Resolve common per-frame depth naming conventions."""
    stem6 = f"{frame_idx:06d}"
    stem4 = f"{frame_idx:04d}"
    candidates = [
        os.path.join(depth_dir, f"{stem6}_depth.npy"),
        os.path.join(depth_dir, f"{stem6}.npy"),
        os.path.join(depth_dir, f"depth_{stem6}.npy"),
        os.path.join(depth_dir, f"{stem4}_depth.npy"),
        os.path.join(depth_dir, f"{stem4}.npy"),
        os.path.join(depth_dir, f"{frame_idx}_depth.npy"),
        os.path.join(depth_dir, f"{frame_idx}.npy"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    glob_candidates = []
    for pattern in (
        os.path.join(depth_dir, f"*{stem6}*depth*.npy"),
        os.path.join(depth_dir, f"*depth*{stem6}*.npy"),
        os.path.join(depth_dir, f"*{stem6}*.npy"),
    ):
        glob_candidates.extend(glob.glob(pattern))

    glob_candidates = sorted(set(glob_candidates))
    if len(glob_candidates) == 1:
        return glob_candidates[0]

    tried = "\n    ".join(candidates)
    if len(glob_candidates) > 1:
        matches = "\n    ".join(glob_candidates)
        raise FileNotFoundError(
            f"Ambiguous depth file for frame {frame_idx} in {depth_dir}; "
            f"exact candidates were missing, glob matched:\n    {matches}"
        )
    raise FileNotFoundError(
        f"Could not find depth file for frame {frame_idx} in {depth_dir}. "
        f"Tried:\n    {tried}"
    )


def backproject(depth_npy, mask_px, T_c2w, scale_s):
    """Back-project masked pixels to 3D world coords using ray-length depth."""
    ys, xs = mask_px
    d_raw = depth_npy[ys, xs].astype(np.float64)
    valid = d_raw > 0
    ys, xs, d_raw = ys[valid], xs[valid], d_raw[valid]

    fx, fy = K_384[0, 0], K_384[1, 1]
    cx, cy = K_384[0, 2], K_384[1, 2]
    rays = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones(len(xs))], axis=1)
    ray_norms = np.linalg.norm(rays, axis=1, keepdims=True)
    rays_unit = rays / ray_norms
    depth_m = d_raw * scale_s
    pts_cam = rays_unit * depth_m[:, None]  # ray-length depth

    pts_h = np.concatenate([pts_cam, np.ones((len(pts_cam), 1))], axis=1)
    pts_world = (T_c2w @ pts_h.T).T[:, :3]
    return pts_world.astype(np.float32)


def voxel_downsample(points, voxel_size):
    """Simple voxel grid downsampling: keep one point per occupied voxel."""
    coords = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    return points[unique_idx]


def project_points(pts_model, T_cam_model):
    """Project Nx3 model-space points to pixel coords. Returns (uv [N,2], valid [N])."""
    pts_cam = (T_cam_model[:3, :3] @ pts_model.T + T_cam_model[:3, 3:4]).T
    valid = pts_cam[:, 2] > 0
    u = np.where(valid, K_384[0, 0] * pts_cam[:, 0] / np.where(valid, pts_cam[:, 2], 1) + K_384[0, 2], -1)
    v = np.where(valid, K_384[1, 1] * pts_cam[:, 1] / np.where(valid, pts_cam[:, 2], 1) + K_384[1, 2], -1)
    in_bounds = (u >= 0) & (u < OUT_SIZE) & (v >= 0) & (v < OUT_SIZE) & valid
    return np.stack([u, v], axis=1).astype(np.float32), in_bounds


def encode_image(rgb_array):
    buf = io.BytesIO()
    Image.fromarray(rgb_array).save(buf, format="PNG")
    return buf.getvalue()


def convert_object(data_dir, out_dir, obj_name, instance_id, caption, frame_rgbs,
                   frame_depths, frame_masks, poses, scale_s):
    """Build and save a ShapeR pkl for one object."""

    # ---- collect 3D points from all frames ----
    all_pts = []
    visible_frames = []
    for i in range(len(poses)):
        mask_px = np.where(frame_masks[i] == instance_id)
        if len(mask_px[0]) < MIN_VISIBLE_PX:
            continue
        pts = backproject(frame_depths[i], mask_px, poses[i], scale_s)
        if len(pts) > 0:
            all_pts.append(pts)
            visible_frames.append(i)

    if len(all_pts) == 0:
        print(f"  [skip] {obj_name}: no visible frames")
        return

    all_pts = np.concatenate(all_pts, axis=0)

    # Voxel downsample, then random subsample to MAX_POINTS
    bbox_diag = np.linalg.norm(all_pts.max(0) - all_pts.min(0))
    voxel = max(bbox_diag / 128, 0.005)
    all_pts = voxel_downsample(all_pts, voxel)
    if len(all_pts) > MAX_POINTS:
        idx = np.random.choice(len(all_pts), MAX_POINTS, replace=False)
        all_pts = all_pts[idx]

    # Center object: model space origin = object centroid
    center = all_pts.mean(axis=0)
    pts_model = all_pts - center  # [N, 3] in model space

    bounds = np.abs(pts_model).max(axis=0).astype(np.float32)
    N = len(pts_model)

    # T_model_world: model->world is just translation by center
    T_model_world = np.eye(4, dtype=np.float32)
    T_model_world[:3, 3] = center

    # ---- build per-frame data ----
    image_data = []
    camera_params = []     # [V, 4]: [fx, fy, cx, cy]
    Ts_camera_model = []   # [V, 4, 4]: T_w2c @ T_model_world (model->cam)
    visible_points_model = []
    object_point_projections = []

    T_w2c_list = [np.linalg.inv(poses[i]).astype(np.float32) for i in range(len(poses))]

    for i in visible_frames:
        T_cam_model = (T_w2c_list[i] @ T_model_world).astype(np.float32)
        uv, in_bounds = project_points(pts_model, T_cam_model)

        vis_pts = pts_model[in_bounds]
        vis_uv = uv[in_bounds]

        image_data.append(encode_image(frame_rgbs[i]))
        camera_params.append([K_384[0, 0], K_384[1, 1], K_384[0, 2], K_384[1, 2]])
        Ts_camera_model.append(T_cam_model)
        visible_points_model.append(torch.tensor(vis_pts, dtype=torch.float32))
        object_point_projections.append(torch.tensor(vis_uv, dtype=torch.float32))

    if len(image_data) == 0:
        print(f"  [skip] {obj_name}: no frames with visible projections")
        return

    pkl = {
        "points_model":             torch.tensor(pts_model, dtype=torch.float32),
        "bounds":                   torch.tensor(bounds, dtype=torch.float32),
        # DPRecon conversion does not provide ShapeR-style uncertainty
        # estimates. Use zero uncertainty so inference thresholding keeps the
        # reconstructed object points.
        "dist_std":                 torch.zeros(N, dtype=torch.float32),
        "inv_dist_std":             torch.zeros(N, dtype=torch.float32),
        "T_model_world":            torch.tensor(T_model_world, dtype=torch.float32),
        "camera_model":             "pinhole",
        "image_data":               image_data,
        "camera_params":            torch.tensor(camera_params, dtype=torch.float32),
        "Ts_camera_model":          torch.tensor(np.stack(Ts_camera_model), dtype=torch.float32),
        "visible_points_model":     visible_points_model,
        "object_point_projections": object_point_projections,
        "caption":                  caption,
        "category":                 caption,
        "is_ariagen2":              False,
    }

    os.makedirs(out_dir, exist_ok=True)
    out_name = f"dprecon__{obj_name}.pkl"
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "wb") as f:
        pickle.dump(pkl, f)
    print(f"  saved {out_name}  ({N} pts, {len(image_data)} views)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Path to DPRecon scan directory")
    parser.add_argument("--depth_dir", default=None, help="Optional directory containing 000000_depth.npy style depth files")
    parser.add_argument("--output_dir", default="data", help="Output directory for pkl files")
    parser.add_argument("--object", default=None, help="Only convert this object name (default: all)")
    args = parser.parse_args()

    data_dir = os.path.expanduser(args.input_dir)
    depth_dir = os.path.expanduser(args.depth_dir) if args.depth_dir else data_dir

    # Load camera poses
    poses = load_poses(os.path.join(data_dir, "traj.txt"))

    # Load scale_s from cameras.npz (same for all frames)
    cam_data = np.load(os.path.join(data_dir, "cameras.npz"))
    scale_s = float(cam_data["scale_mat_0"][0, 0])
    print(f"scale_s = {scale_s:.5f}")

    # Load object descriptions
    with open(os.path.join(data_dir, "replica_scan1_prompt_geometry.json")) as f:
        prompts = json.load(f)
    with open(os.path.join(data_dir, "instance_id.json")) as f:
        instance_ids = json.load(f)

    # Load all frame images, depths, masks once
    frame_rgbs, frame_depths, frame_masks = [], [], []
    for i in range(10):
        frame_rgbs.append(np.array(Image.open(
            os.path.join(data_dir, f"{i:06d}_rgb.png")).convert("RGB")))
        frame_depths.append(np.load(resolve_depth_path(depth_dir, i)))
        mask_img = np.array(Image.open(
            os.path.join(data_dir, "instance_mask", f"{i:06d}.png")))
        frame_masks.append(mask_img[:, :, 0] if mask_img.ndim == 3 else mask_img)

    # Convert each object
    for obj_idx, (obj_name, inst_id) in enumerate(instance_ids.items()):
        if args.object and obj_name != args.object:
            continue
        caption = prompts.get(str(obj_idx + 1), obj_name)
        print(f"Converting {obj_name} (id={inst_id}): {caption}")
        convert_object(
            data_dir, args.output_dir,
            obj_name, inst_id, caption,
            frame_rgbs, frame_depths, frame_masks,
            poses, scale_s,
        )


if __name__ == "__main__":
    main()
