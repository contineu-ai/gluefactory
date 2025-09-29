import numpy as np
import torch
from scipy.ndimage import map_coordinates

# --- Coordinate System Helpers ---
# We use a standard Y-up coordinate system
# phi: longitude, from -pi to pi
# theta: latitude, from -pi/2 to pi/2

def spherical_to_cartesian(kpts_sph_np):
    """Converts spherical coordinates (phi, theta) to 3D Cartesian (x, y, z)."""
    phi = kpts_sph_np[:, 0]    # Longitude
    theta = kpts_sph_np[:, 1]  # Latitude
    
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta)
    z = np.cos(theta) * np.cos(phi)
    return np.stack([x, y, z], axis=-1)

def cartesian_to_spherical(xyz_np):
    """Converts 3D Cartesian (x, y, z) to spherical (phi, theta)."""
    x, y, z = xyz_np[:, 0], xyz_np[:, 1], xyz_np[:, 2]
    
    phi = np.arctan2(x, z)
    theta = np.arcsin(y)
    return np.stack([phi, theta], axis=-1)

def standard_spherical_to_pixel(kpts_sph_np, W, H):
    """
    Converts standard spherical coordinates to pixel coordinates.
    phi: longitude [-pi, pi] -> x [0, W]
    theta: latitude [-pi/2, pi/2] -> y [0, H]
    """
    phi = kpts_sph_np[:, 0]
    theta = kpts_sph_np[:, 1]

    # Normalize phi to [0, 1] and theta to [0, 1] and Scale to pixel coordinates
    px = (phi / (2 * np.pi) + 0.5) * (W - 1) - 0.5
    py = (-theta / np.pi + 0.5) * (H - 1) - 0.5
    
    return np.stack([px, py], axis=-1)

def standard_pixel_to_spherical(kpts_np, W, H):
    """
    Converts standard pixel coordinates to spherical coordinates.
    phi: longitude [-pi, pi] <- x [0, W]
    theta: latitude [-pi/2, pi/2] <- y [0, H]
    """
    px = kpts_np[:, 0]
    py = kpts_np[:, 1]

    # Normalize phi to [0, 1] and theta to [0, 1] and Scale to pixel coordinates
    phi = ((px + 0.5) / (W - 1) - 0.5) * 2 * np.pi
    theta = (0.5 - (py + 0.5) / (H - 1))*np.pi
    
    return np.stack([phi, theta], axis=-1)

def rotation_matrix(yaw, pitch, roll):
    """
    Creates a rotation matrix for yaw, pitch, roll in degrees.
    Yaw: rotation around Y-axis
    Pitch: rotation around X-axis
    Roll: rotation around Z-axis
    """
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)

    # Yaw matrix (rotation around Y-axis)
    R_yaw = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])

    # Pitch matrix (rotation around X-axis)
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])

    # Roll matrix (rotation around Z-axis)
    R_roll = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1]
    ])

    # Combine rotations (Roll -> Pitch -> Yaw)
    R = R_yaw @ R_pitch @ R_roll

    return R

def rotate_image(image_np, yaw, pitch, roll):
    """
    Rotates an equirectangular image
    
    Args:
        image_np (torch.tensor): The input image (C, H, W).
        yaw, pitch, roll (float): Rotation angles in degrees.
        
    Returns:
        np.ndarray: The rotated image.
    """
    
    # 1. Convert image to numpy array if it's a tensor
    image_np = image_np.cpu().numpy() if isinstance(image_np, torch.Tensor) else image_np
    image_np = image_np.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C) for processing
    height, width, channels = image_np.shape   

    # 2. Create a grid of coordinates for the output image
    # u corresponds to longitude (phi), v to latitude (theta)
    u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
    
    # 3. Convert pixel coordinates to spherical coordinates (longitude, latitude)
    # Phi (longitude) ranges from -pi to pi
    # Theta (latitude) ranges from -pi/2 to pi/2
    phi   =  ( (u_grid + 0.5) / (width - 1) - 0.5) * 2 * np.pi
    theta =  ((-v_grid - 0.5) / (height - 1) + 0.5) * np.pi

    # 4. Convert spherical coordinates to 3D Cartesian coordinates
    # Y-up coordinate system: Y is vertical, X is right, Z is forward
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta)
    z = np.cos(theta) * np.cos(phi)

    # 5. Create rotation matrix from yaw, pitch, roll
    # Create rotation matrices (in radians)
    # We apply the INVERSE rotation to the coordinates

    R = rotation_matrix(yaw, pitch, roll)

    # 6. Apply the rotation to the 3D coordinates
    # Reshape for matrix multiplication
    xyz = np.stack((x, y, z), axis=-1)
    xyz_rotated = xyz.reshape(-1, 3) @ R.T
    xyz_rotated = xyz_rotated.reshape(height, width, 3)
    
    x_rot, y_rot, z_rot = xyz_rotated[..., 0], xyz_rotated[..., 1], xyz_rotated[..., 2]

    # 7. Convert rotated 3D Cartesian coordinates back to spherical
    phi_src = np.arctan2(x_rot, z_rot)
    theta_src = np.arcsin(y_rot)

    # 8. Convert spherical coordinates back to source pixel coordinates
    u_src = np.clip((phi_src / (2 * np.pi) + 0.5) * (width - 1), 0, width - 1)
    v_src = np.clip((-theta_src / np.pi + 0.5) * (height - 1), 0, height - 1)

    # 9. Sample the source image using the calculated coordinates
    # map_coordinates is a powerful function for this. It handles interpolation.
    # The first argument is the coordinates for each dimension: (v_coords, u_coords, channel_coords)
    rotated_img_arr = np.zeros_like(image_np)

    for c in range(channels):
        rotated_img_arr[..., c] = map_coordinates(
            image_np[..., c],
            [v_src, u_src],
            order=1,
            mode='wrap' # Use 'wrap' for longitude, 'reflect' or 'nearest' for latitude could also work
        )

    return rotated_img_arr

def rotate_image_batch(image_np, yaw, pitch, roll):
    """
    Rotates a batch of equirectangular images on the GPU using torch.grid_sample.

    Args:
        images_torch (torch.Tensor): Batch of images (B, C, H, W) on the GPU.
        yaws (torch.Tensor): Batch of yaw angles in degrees (B,).
        pitches (torch.Tensor): Batch of pitch angles in degrees (B,).
        rolls (torch.Tensor): Batch of roll angles in degrees (B,).

    Returns:
        torch.Tensor: The batch of rotated images.
    """
    
    # B, C, H, W = images_torch.shape
    # device = images_torch.device

    # 1. Convert image to numpy array if it's a tensor
    image_np = image_np.cpu().numpy() if isinstance(image_np, torch.Tensor) else image_np
    image_np = image_np.transpose(1, 2, 0)  # Convert from (B, C, H, W) to (B, H, W, C) for processing
    height, width, channels = image_np.shape   

    # 2. Create a grid of coordinates for the output image
    # u corresponds to longitude (phi), v to latitude (theta)
    u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
    
    # 3. Convert pixel coordinates to spherical coordinates (longitude, latitude)
    # Phi (longitude) ranges from -pi to pi
    # Theta (latitude) ranges from -pi/2 to pi/2
    phi   =  ( (u_grid + 0.5) / (width - 1) - 0.5) * 2 * np.pi
    theta =  ((-v_grid - 0.5) / (height - 1) + 0.5) * np.pi

    # 4. Convert spherical coordinates to 3D Cartesian coordinates
    # Y-up coordinate system: Y is vertical, X is right, Z is forward
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta)
    z = np.cos(theta) * np.cos(phi)

    # 5. Create rotation matrix from yaw, pitch, roll
    # Create rotation matrices (in radians)
    # We apply the INVERSE rotation to the coordinates

    R = rotation_matrix(yaw, pitch, roll)

    # 6. Apply the rotation to the 3D coordinates
    # Reshape for matrix multiplication
    xyz = np.stack((x, y, z), axis=-1)
    xyz_rotated = xyz.reshape(-1, 3) @ R.T
    xyz_rotated = xyz_rotated.reshape(height, width, 3)
    
    x_rot, y_rot, z_rot = xyz_rotated[..., 0], xyz_rotated[..., 1], xyz_rotated[..., 2]

    # 7. Convert rotated 3D Cartesian coordinates back to spherical
    phi_src = np.arctan2(x_rot, z_rot)
    theta_src = np.arcsin(y_rot)

    # 8. Convert spherical coordinates back to source pixel coordinates
    u_src = np.clip((phi_src / (2 * np.pi) + 0.5) * (width - 1), 0, width - 1)
    v_src = np.clip((-theta_src / np.pi + 0.5) * (height - 1), 0, height - 1)

    # 9. Sample the source image using the calculated coordinates
    # map_coordinates is a powerful function for this. It handles interpolation.
    # The first argument is the coordinates for each dimension: (v_coords, u_coords, channel_coords)
    rotated_img_arr = np.zeros_like(image_np)

    for c in range(channels):
        rotated_img_arr[..., c] = map_coordinates(
            image_np[..., c],
            [v_src, u_src],
            order=1,
            mode='wrap' # Use 'wrap' for longitude, 'reflect' or 'nearest' for latitude could also work
        )

    return rotated_img_arr    

def rotate_keypoints(kpts_sph_np, yaw, pitch, roll, inverse=False):
    """
    Rotates keypoints in spherical coordinates.
    
    Args:
        kpts_sph_np (np.ndarray): Keypoints in spherical coordinates (phi, theta).
        yaw, pitch, roll (float): Rotation angles in degrees.
        
    Returns:
        np.ndarray: Rotated keypoints in spherical coordinates.
    """
    # 1. Convert spherical coordinates to Cartesian
    xyz = spherical_to_cartesian(kpts_sph_np)

    # 2. Create rotation matrix
    R = rotation_matrix(yaw, pitch, roll)

    # 3. Apply rotation
    if inverse:
        xyz_rotated = xyz @ R.T
    else:
        xyz_rotated = xyz @ R

    # 4. Convert back to spherical coordinates
    kpts_rotated_sph = cartesian_to_spherical(xyz_rotated)

    return kpts_rotated_sph