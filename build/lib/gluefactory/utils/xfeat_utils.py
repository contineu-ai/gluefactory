# --- xfeat import ---
import sys
sys.path.append("/mnt/d/code/accelerated_features")
from modules.xfeat import XFeat

import numpy as np

xfeat = XFeat()

# Dicemap is a 4x3 grid of faces
DICEMAP_W, DICEMAP_H = 2048, 1536
FACE_W = DICEMAP_W // 4  # 2048 / 4 = 512
FACE_H = DICEMAP_H // 3  # 1536 / 3 = 512

# Equirectangular image dimensions
EQUI_W, EQUI_H = 2048, 1024

# Layout of faces on the dicemap grid (col, row)
# This is the same as your FACE_LAYOUT
FACE_LAYOUT = {
    'R': (2, 1),  # Right
    'L': (0, 1),  # Left
    'U': (1, 0),  # Up
    'D': (1, 2),  # Down
    'F': (1, 1),  # Front
    'B': (3, 1),  # Back
}

# We need an inverted mapping for easy lookup from grid position to face name
INVERTED_FACE_LAYOUT = {v: k for k, v in FACE_LAYOUT.items()}


def dicemap_to_face_coords(dicemap_x, dicemap_y):
    """
    Step 1: Convert dicemap coordinate to a specific face and its local coordinates.
    (This function is correct and remains unchanged)
    """
    if not (0 <= dicemap_x < DICEMAP_W and 0 <= dicemap_y < DICEMAP_H):
        return None, None, None
        
    grid_x = dicemap_x // FACE_W
    grid_y = dicemap_y // FACE_H

    face_name = INVERTED_FACE_LAYOUT.get((grid_x, grid_y))
    
    if face_name is None:
        return None, None, None

    face_x = dicemap_x % FACE_W
    face_y = dicemap_y % FACE_H
    
    return face_name, face_x, face_y


def face_to_xyz(face_name, face_x, face_y):
    """
    Step 2 & 3: CORRECTED conversion from face coordinates to a 3D vector.
    This function now accurately reverses the logic from your `xyzcube` function.
    """
    # Normalize face_x and face_y to the range [-0.5, 0.5]
    # This matches the `rng = np.linspace(-0.5, 0.5, ...)` from your code
    s = (face_x / (FACE_W - 1)) - 0.5
    t = (face_y / (FACE_H - 1)) - 0.5

    # From your code `grid = np.stack(np.meshgrid(rng, -rng), -1)`
    # The x-component of the grid maps to `s`.
    # The y-component of the grid maps to `-t`.

    if face_name == 'F':
        # Forward: x,y,z = grid.x, grid.y, 0.5
        # Reverse:
        x, y, z = s, -t, 0.5
    elif face_name == 'B':
        # Forward: x,y,z = grid_b.x, grid_b.y, -0.5
        # grid_b flips the x-axis, so grid_b.x is -s
        # Reverse:
        x, y, z = -s, -t, -0.5
    elif face_name == 'R':
        # Forward: x,y,z = 0.5, grid_r.y, grid_r.x
        # grid_r flips the x-axis, so grid_r.x is -s
        # Reverse:
        x, y, z = 0.5, -t, -s
    elif face_name == 'L':
        # Forward: x,y,z = -0.5, grid.y, grid.x
        # Reverse:
        x, y, z = -0.5, -t, s
    elif face_name == 'U':
        # Forward: x,y,z = grid_u.x, 0.5, grid_u.y
        # grid_u flips the y-axis, making grid_u.y map directly to `t`
        # Reverse:
        x, y, z = s, 0.5, t
    elif face_name == 'D':
        # Forward: x,y,z = grid.x, -0.5, grid.y
        # grid.y maps to `-t`
        # Reverse:
        x, y, z = s, -0.5, -t
    else:
        raise ValueError("Invalid face name")

    # The resulting (x,y,z) is on the cube. Normalize to project to sphere.
    xyz_vec = np.array([x, y, z])
    return xyz_vec / np.linalg.norm(xyz_vec)


def xyz_to_uv(xyz):
    """
    Step 4: Convert 3D cartesian coordinates to spherical coordinates (u, v).
    (This function is correct and remains unchanged)
    """
    x, y, z = xyz
    u = np.arctan2(x, z)
    c = np.sqrt(x**2 + z**2)
    v = np.arctan2(y, c)
    return u, v


def uv_to_equirectangular(u, v):
    """
    Step 5: Convert spherical (u, v) coordinates to equirectangular pixel coordinates.
    (This function is correct and remains unchanged)
    """
    x = (u / (2 * np.pi) + 0.5) * EQUI_W - 0.5
    y = (-v / np.pi + 0.5) * EQUI_H - 0.5
    return x, y


# --- Main Conversion Function ---

def dicemap_to_equirectangular_keypoints(dicemap_x, dicemap_y):
    """
    Converts a single (x, y) coordinate from a dicemap image to its
    corresponding (x', y') coordinate in an equirectangular image.

    Args:
        dicemap_x (int): The x-coordinate on the (2048, 1536) dicemap.
        dicemap_y (int): The y-coordinate on the (2048, 1536) dicemap.

    Returns:
        tuple: (equi_x, equi_y) coordinates, or (None, None) if the input
               is on an empty part of the dicemap.
    """
    # 1. Dicemap coordinate to Face coordinate
    face_name, face_x, face_y = dicemap_to_face_coords(dicemap_x, dicemap_y)
    if face_name is None:
        return None, None, None, None

    # 2. Face coordinate to 3D Sphere coordinate
    xyz_sphere = face_to_xyz(face_name, face_x, face_y)
    
    # 3. 3D Sphere coordinate to (u,v)
    u, v = xyz_to_uv(xyz_sphere)

    # 4. (u,v) to Equirectangular coordinate
    equi_x, equi_y = uv_to_equirectangular(u, v)
    
    return equi_x, equi_y, u, v


def generate_keypoints(image, num_keypoints=4096):
    """
    Detects keypoints, descriptors, and scores using XFeat, converts keypoint
    coordinates to spherical, and filters out any invalid keypoints along with
    their corresponding descriptors and scores.

    Args:
        image: The input image for feature detection.
        num_keypoints (int): The maximum number of keypoints to detect.

    Returns:
        A tuple containing:
        - keypointCoords (np.ndarray): An array of shape (N, 2) of valid spherical
          coordinates [phi, theta] as float32. N <= num_keypoints.
        - keypointDescriptors (np.ndarray): An array of shape (N, D) of descriptors
          corresponding to the valid keypoints. D is the descriptor dimension.
        - keypointScores (np.ndarray): An array of shape (N,) of scores
          corresponding to the valid keypoints.
    """
    # 1. Detect features and get PyTorch tensors
    # The [0] assumes a batch size of 1
    output = xfeat.detectAndCompute(image, num_keypoints)[0]

    # 2. Convert tensors to NumPy arrays for easier processing
    keypoints_d = output['keypoints'].cpu().numpy()
    descriptors_d = output['descriptors'].cpu().numpy()
    scores_d = output['scores'].cpu().numpy()

    # 3. Initialize lists to store only the valid, filtered data
    valid_spherical_keypoints = []
    valid_descriptors = []
    valid_scores = []
    # Optional: if you also need the equirectangular keypoints
    # valid_equirect_keypoints = []

    # 4. Iterate and filter simultaneously
    # We zip the keypoints, descriptors, and scores together so they are processed
    # in lockstep.
    for kp, descriptor, score in zip(keypoints_d, descriptors_d, scores_d):
        # Unpack the keypoint from the dicemap format
        x_d, y_d = kp
        
        # Convert coordinates
        equi_x, equi_y, phi, theta = dicemap_to_equirectangular_keypoints(x_d, y_d)

        # Check if the conversion was successful. If not, this keypoint
        # and its associated descriptor and score are skipped.
        if phi is None or theta is None:
            continue

        # If the keypoint is valid, add it and its corresponding descriptor
        # and score to our lists.
        valid_spherical_keypoints.append([phi, theta])
        valid_descriptors.append(descriptor)
        valid_scores.append(score)
        # Optional:
        # valid_equirect_keypoints.append([equi_x, equi_y])

    # 5. Convert lists of valid data to NumPy arrays with the correct types
    if not valid_spherical_keypoints:
        # Handle the edge case where no valid keypoints were found
        descriptor_dim = descriptors_d.shape[1] if descriptors_d.ndim > 1 else 0
        return (
            np.empty((0, 2), dtype='float32'),
            np.empty((0, descriptor_dim), dtype=descriptors_d.dtype),
            np.empty((0,), dtype=scores_d.dtype)
        )

    keypointCoords = np.array(valid_spherical_keypoints, dtype='float32')
    keypointDescriptors = np.array(valid_descriptors)
    keypointScores = np.array(valid_scores)

    return (keypointCoords, keypointDescriptors, keypointScores)