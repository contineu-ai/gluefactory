import cv2
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
from copy import deepcopy
import logging

def read_depth(p):
    d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if d.ndim == 3: d = d[...,0]
    return d.astype(np.float32)


def get_c2w_and_w2c_matrix(filepath):
    """Reads a pose from a specific .dat file format.

    Each file contains the rotation matrix R stored as a 3 x 3 matrix that, following convention, 
    encodes the transformation from world to camera coordinate system. However, here the vector t 
    does not follow convention and represents the camera position in world coordinate system. This 
    is convenient to compute distances between cameras and determine neighborhood.

    1. R is a 3x3 matrix from world-to-camera. This means P_camera = R @ P_world.
    2. t (as stored in the file) represents the camera position in the world coordinate system. 
    This means t is what we typically call the camera center, C.

    Args:
        filepath: Path to the .dat file.

    Returns:
        A 4x4 Camera-to-World (C2W) matrix.
    """

    # Load the 3x3 rotation matrix, skipping the first 2 header lines
    # and reading only the 3 subsequent lines.
    R_w2c = np.loadtxt(filepath, skiprows=2, max_rows=3)
    
    # Load the 3x1 translation vector, skipping all lines before it.
    # (2 header lines + 3 matrix rows + 't' marker = 6 lines)
    C_world = np.loadtxt(filepath, skiprows=6)

    # --- Convert [R_w2c | C] to a 4x4 C2W matrix ---
    # The rotation part of a C2W matrix is the inverse (transpose) of a W2C rotation.
    R_c2w = R_w2c.T
    
    # The translation part of a C2W matrix is simply the camera center C.
    t_c2w = C_world
    
    # Assemble the 4x4 matrix
    pose_c2w = np.eye(4)
    pose_c2w[:3, :3] = R_c2w
    pose_c2w[:3, 3] = t_c2w

    pose_w2c = np.eye(4)
    pose_w2c[:3, :3] = R_w2c
    pose_w2c[:3, 3] = -R_w2c.dot(t_c2w)

    return pose_c2w, pose_w2c


def unproject_spherical(uv, d, w, h):
    u = uv[:, 0].astype(float)
    v = uv[:, 1].astype(float)
    r = d.squeeze().astype(float)
    phi = (v + 0.5) * np.pi / h
    theta = (1.0 - (u + 0.5) / w) * (2.0 * np.pi)
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)
    return np.stack([x, y, z], axis=1)


def project_spherical(pts, w, h):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    d = np.linalg.norm(pts, axis=-1)
    d_safe = np.clip(d, a_min=1e-8, a_max=None)  # Prevent division by zero

    phi = np.arccos(np.clip(z/d_safe, a_min=-1.0, a_max=1.0)) 
    theta = np.arctan2(y, x) # Shift to [0, 2*pi]
    theta = np.remainder(theta, 2.0 * np.pi)

    u = (1.0 - theta / (2.0 * np.pi)) * w - 0.5
    v = (phi / np.pi) * h - 0.5
    
    uv = np.stack([u, v], axis=1)
    return uv, d


class EmptyTensorError(Exception):
    pass


def interpolate_depth(pos, depth):
    """Interpolates depth values for 2D points using bilinear interpolation."""
    # Ensure pos is 2xN and convert to integer indices
    pos = pos.T[[1, 0]]

    h, w = depth.shape
    
    i = pos[0, :].astype(float)
    j = pos[1, :].astype(float)

    # Valid corners and indices
    i_top_left = np.floor(i).astype(int)
    j_top_left = np.floor(j).astype(int)
    valid_top_left = np.logical_and(i_top_left >= 0, j_top_left >= 0)

    i_top_right = np.floor(i).astype(int)
    j_top_right = np.ceil(j).astype(int)
    valid_top_right = np.logical_and(i_top_right >= 0, j_top_right < w)

    i_bottom_left = np.ceil(i).astype(int)
    j_bottom_left = np.floor(j).astype(int)
    valid_bottom_left = np.logical_and(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = np.ceil(i).astype(int)
    j_bottom_right = np.ceil(j).astype(int)
    valid_bottom_right = np.logical_and(i_bottom_right < h, j_bottom_right < w)

    valid_corners = np.all(
        [valid_top_left, valid_top_right, valid_bottom_left, valid_bottom_right], axis=0)

    ids = np.arange(pos.shape[1])
    ids_valid_corners = ids[valid_corners]
    
    if ids_valid_corners.size == 0:
        raise EmptyTensorError

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]
    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]
    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]
    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]
    
    # Check depth validity
    valid_depth = np.all(
        [
            depth[i_top_left, j_top_left] > 0,
            depth[i_top_right, j_top_right] > 0,
            depth[i_bottom_left, j_bottom_left] > 0,
            depth[i_bottom_right, j_bottom_right] > 0
        ],
        axis=0,
    )

    ids = ids_valid_corners[valid_depth]
    ids_valid_depth = deepcopy(ids)

    if ids.size == 0:
        raise EmptyTensorError

    i = i[ids]
    j = j[ids]

    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]
    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]
    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]
    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]

    # Interpolation
    dist_i = i - i_top_left
    dist_j = j - j_top_left
    w_top_left = (1 - dist_i) * (1 - dist_j)
    w_top_right = (1 - dist_i) * dist_j
    w_bottom_left = dist_i * (1 - dist_j)
    w_bottom_right = dist_i * dist_j

    interpolated_depth = (w_top_left * depth[i_top_left, j_top_left] +
                          w_top_right * depth[i_top_right, j_top_right] +
                          w_bottom_left * depth[i_bottom_left, j_bottom_left] +
                          w_bottom_right * depth[i_bottom_right, j_bottom_right])

    pos_valid = pos[:, ids]
    pos_valid = pos_valid[[1, 0]].T

    return [interpolated_depth, pos_valid, ids, ids_valid_corners, ids_valid_depth]


def warp_points3d(points3d0: np.ndarray, pose01: np.ndarray) -> np.ndarray:
    """Warps 3D points using a SE3 pose."""
    points3d0_homo = np.concatenate([points3d0, np.ones((points3d0.shape[0], 1))], axis=1)
    points3d01_homo = np.einsum('jk,nk->nj', pose01, points3d0_homo)
    return points3d01_homo[:, 0:3]


def warp_se3_spherical(kpts0: np.ndarray, params: dict) -> tuple:
    """Warps 2D keypoints from one spherical image to another using 3D transformation and validation."""
    pose01 = params['pose01']
    depth0 = params['depth0'].squeeze()
    depth1 = params['depth1'].squeeze()
    W, H = params['width'], params['height']
    abs_tol = params.get('abs_tol', 0.05)
    rel_tol = params.get('rel_tol', 0.02)

    try:
        # 1) Get depth for keypoints
        z0, k0v, ids0, _, _ = interpolate_depth(kpts0, depth0)
    except EmptyTensorError:
        logging.warning("No valid keypoints after img0 depth check.")
        return kpts0, kpts0, np.empty(0, dtype=np.long), np.empty(0, dtype=np.long)

    # 2) Unproject -> warp -> project
    pts3d_0 = unproject_spherical(k0v, z0, W, H)
    pts3d_1 = warp_points3d(pts3d_0, pose01)
    uv1_pred, z1_proj = project_spherical(pts3d_1, W, H)

    try:
        # 3) Depth check img1
        z1i, k1v, ids1, _, _ = interpolate_depth(uv1_pred, depth1)
    except EmptyTensorError:
        logging.warning("All warped keypoints invalid in img1.")
        return kpts0, uv1_pred, ids0, ids1
        
    # 4) Occlusion check (depth consistency)
    abs_diff = np.abs(z1_proj[ids1] - z1i)
    rel_diff = abs_diff / np.clip(z1i, a_min=1e-6, a_max=None)
    mask = (abs_diff < abs_tol) & (rel_diff < rel_tol)
    
    # Filter points based on the mask
    final_ids = ids0[ids1][mask]
    k0_final = k0v[ids1][mask]
    k1_final = k1v[mask]
    
    # 5) Handle spherical wrap-around
    u0n = k0_final[:, 0] / (W - 1)
    u1n = k1_final[:, 0] / (W - 1)
    dn = np.remainder(u1n - u0n + 0.5, 1.0) - 0.5
    uc = np.remainder(u0n + dn, 1.0)
    k1_final[:, 0] = uc * (W - 1)
    k1_final[:, 1] = np.clip(k1_final[:, 1], a_min=0, a_max=H - 1)
    
    return k0_final, k1_final, final_ids, np.empty(0, dtype=np.int64)