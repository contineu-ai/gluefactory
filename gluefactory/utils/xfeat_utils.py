# --- xfeat import ---
import sys
sys.path.append("/data/code/accelerated_features")
from modules.xfeat import XFeat
import matplotlib.pyplot as plt
import numpy as np

# --- Module-level constants ---
# These layouts are fixed and don't depend on image size,
# so they can be defined here as constants.
FACE_LAYOUT = {
    'R': (2, 1),  # Right
    'L': (0, 1),  # Left
    'U': (1, 0),  # Up
    'D': (1, 2),  # Down
    'F': (1, 1),  # Front
    'B': (3, 1),  # Back
}

INVERTED_FACE_LAYOUT = {v: k for k, v in FACE_LAYOUT.items()}

xfeat = XFeat()

# --- Refactored Conversion Functions ---

def dicemap_to_face_coords(dicemap_x, dicemap_y, dims):
    """
    Step 1: Convert dicemap coordinate to a specific face and its local coordinates.
    """
    if not (0 <= dicemap_x < dims['DICEMAP_W'] and 0 <= dicemap_y < dims['DICEMAP_H']):
        return None, None, None
        
    grid_x = int(dicemap_x // dims['FACE_W'])
    grid_y = int(dicemap_y // dims['FACE_H'])

    face_name = INVERTED_FACE_LAYOUT.get((grid_x, grid_y))
    
    if face_name is None:
        return None, None, None

    face_x = dicemap_x % dims['FACE_W']
    face_y = dicemap_y % dims['FACE_H']
    
    return face_name, face_x, face_y


def face_to_xyz(face_name, face_x, face_y, dims):
    """
    Step 2 & 3: Convert face coordinates to a 3D vector on a unit sphere.
    """
    # Normalize face_x and face_y to the range [-0.5, 0.5]
    s = (face_x / (dims['FACE_W'] - 1)) - 0.5
    t = (face_y / (dims['FACE_H'] - 1)) - 0.5

    if face_name == 'F':
        x, y, z = s, -t, 0.5
    elif face_name == 'B':
        x, y, z = -s, -t, -0.5
    elif face_name == 'R':
        x, y, z = 0.5, -t, -s
    elif face_name == 'L':
        x, y, z = -0.5, -t, s
    elif face_name == 'U':
        x, y, z = s, 0.5, t
    elif face_name == 'D':
        x, y, z = s, -0.5, -t
    else:
        raise ValueError("Invalid face name")

    # Normalize the cube coordinate to project it onto the unit sphere
    xyz_vec = np.array([x, y, z])
    return xyz_vec / np.linalg.norm(xyz_vec)


def xyz_to_uv(xyz):
    """
    Step 4: Convert 3D cartesian coordinates to spherical coordinates (phi, theta).
    """
    x, y, z = xyz
    phi = np.arctan2(x, z)  # Azimuthal angle (longitude)
    theta = np.arcsin(y)  # Polar angle (latitude)
    return phi, theta


def uv_to_equirectangular(phi, theta, dims):
    """
    Step 5: Convert spherical (phi, theta) coordinates to equirectangular pixel coordinates.
    """
    x = (phi / (2 * np.pi) + 0.5) * dims['EQUI_W'] - 0.5
    y = (-theta / np.pi + 0.5) * dims['EQUI_H'] - 0.5
    return x, y


def dicemap_to_equirectangular_keypoints(dicemap_x, dicemap_y, dims):
    """
    Converts a single (x, y) coordinate from a dicemap image to its
    corresponding (x', y') coordinate in an equirectangular image and spherical (u,v).
    """
    # 1. Dicemap coordinate to Face coordinate
    face_name, face_x, face_y = dicemap_to_face_coords(dicemap_x, dicemap_y, dims)
    if face_name is None:
        return None, None, None, None

    # 2. Face coordinate to 3D Sphere coordinate
    xyz_sphere = face_to_xyz(face_name, face_x, face_y, dims)
    
    # 3. 3D Sphere coordinate to (u,v)
    phi, theta = xyz_to_uv(xyz_sphere)

    # 4. (u,v) to Equirectangular coordinate
    equi_x, equi_y = uv_to_equirectangular(phi, theta, dims)
    
    return equi_x, equi_y, phi, theta


# --- Main Orchestration Function ---

def generate_keypoints(image, num_keypoints=4096):
    """
    Detects keypoints from a dicemap image, converts their coordinates to
    spherical, and returns the valid keypoints with their descriptors and scores.
    """
    # 0. Dynamically calculate dimensions from the input dicemap image
    dicemap_h, dicemap_w = image.shape[:2]
    
    # The dicemap is a 4x3 grid of square faces
    face_w = dicemap_w // 4
    face_h = dicemap_h // 3

    # Store dimensions in a dictionary to pass to helper functions
    dims = {
        "DICEMAP_W": dicemap_w,
        "DICEMAP_H": dicemap_h,
        "FACE_W": face_w,
        "FACE_H": face_h,
        "EQUI_W": face_w * 4,
        "EQUI_H": face_w * 2, # Equirectangular is 2:1 aspect ratio
    }

    # 1. Detect features
    output = xfeat.detectAndCompute(image, num_keypoints)[0]

    # 2. Convert tensors to NumPy arrays
    keypoints_d = output['keypoints'].cpu().numpy()
    descriptors_d = output['descriptors'].cpu().numpy()
    scores_d = output['scores'].cpu().numpy()

    # 3. Initialize lists to store valid, filtered data
    valid_spherical_keypoints = []
    valid_descriptors = []
    valid_scores = []
    # valid_equirect_keypoints = [] # Optional, for visualization

    # 4. Iterate, convert, and filter keypoints
    for kp, descriptor, score in zip(keypoints_d, descriptors_d, scores_d):
        x_d, y_d = kp
        
        # Convert coordinates, passing the calculated dimensions
        equi_x, equi_y, phi, theta = dicemap_to_equirectangular_keypoints(x_d, y_d, dims)

        # Skip keypoints that fall on empty parts of the dicemap
        if phi is None or theta is None:
            continue

        # Append valid data to lists
        valid_spherical_keypoints.append([phi, theta])
        valid_descriptors.append(descriptor)
        valid_scores.append(score)
        # valid_equirect_keypoints.append([equi_x, equi_y])

    # 5. Convert lists to final NumPy arrays
    if not valid_spherical_keypoints:
        # Handle case where no valid keypoints were found
        desc_dim = descriptors_d.shape[1] if descriptors_d.ndim > 1 else 0
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, desc_dim), dtype=descriptors_d.dtype),
            np.empty((0,), dtype=scores_d.dtype)
        )

    keypointCoords = np.array(valid_spherical_keypoints, dtype=np.float32)
    keypointDescriptors = np.array(valid_descriptors)
    keypointScores = np.array(valid_scores)

    return (keypointCoords, keypointDescriptors, keypointScores)