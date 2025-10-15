import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt

# --- Assumed GlueFactory Utilities ---
# These functions are critical and are assumed to be available from your environment,
# as they were in the script you provided.
GLUE_FACTORY_ROOT = Path(__file__).resolve().parent.parent # Example if script is in glue_factory_root/scripts/
sys.path.insert(0, str(GLUE_FACTORY_ROOT))

from gluefactory.utils.equirectangular_utils import equirectangular_to_dicemap
from gluefactory.utils.xfeat_utils import generate_keypoints
from gluefactory.utils.spherical_utils import standard_spherical_to_pixel, standard_pixel_to_spherical, spherical_to_cartesian
from gluefactory.utils.spherecraft_utils import read_depth, get_c2w_and_w2c_matrix, warp_se3_spherical

# Third-party libraries (ensure these are installed)
import numpy as np
import torch
import cv2
from joblib import Parallel, delayed

# Configure OpenCV to read EXR files
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ==============================================================================
# DATA LOADING UTILITIES
# ==============================================================================

def read_pairs_from_txt(filepath: Path) -> List[Tuple[str, str]]:
    """Reads image pairs from a text file.
    
    Assumes each line contains two image filenames separated by whitespace.
    Example line: '00000004.png 00000040.png'
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Pair file not found at {filepath}")
    
    pairs = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split("/")[-1].split(".")[0].split("_")
            pairs.append((parts[0], parts[1]))

    logging.info(f"Loaded {len(pairs)} pairs from {filepath.name}")
    return pairs

def plot_img(image, kpts_spherical, color=(255, 0, 255), radius=1, thickness=5):
    """
    Draws keypoints on an image and displays it.

    Args:
        image (np.ndarray): The input image in BGR format (as read by cv2).
        kpts_spherical (np.ndarray): Keypoints in spherical coordinates (phi, theta).
        color (tuple): BGR color for the keypoints.
        radius (int): Radius of the circles representing keypoints.
        thickness (int): Thickness of the circle outline.
    """
    h, w = image.shape[:2]

    # Convert spherical keypoints to pixel coordinates in the image
    pixel_coords = standard_spherical_to_pixel(kpts_spherical, w, h)

    # Create a copy of the image to avoid modifying the original
    img_with_kpts = image.copy()

    # Draw each keypoint as a circle on the image
    for point in pixel_coords:
        # Get integer coordinates for drawing
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(img_with_kpts, (x, y), radius, color, thickness)

    # --- Display the image using Matplotlib ---
    # Convert the image from BGR (OpenCV's default) to RGB for correct color display
    img_rgb = cv2.cvtColor(img_with_kpts, cv2.COLOR_BGR2RGB)

    # Create a plot to show the image
    plt.figure(figsize=(16, 8))
    plt.imshow(img_rgb)
    plt.title("Image with Plotted Keypoints")
    plt.axis('off')  # Hide the axes for a cleaner look
    plt.show()

# ==============================================================================
# CORE LOGIC
# ==============================================================================

def generate_sfm_groundtruth(
    kpts_A_sph: np.ndarray,
    kpts_B_sph: np.ndarray,
    depth_map_A: np.ndarray,
    depth_map_B: np.ndarray,
    pose_A_c2w: Dict[str, np.ndarray],  
    pose_B_w2c: Dict[str, np.ndarray],
    angle_threshold_degrees: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Generates ground truth matches using SfM poses and a depth map.
    """

    if kpts_A_sph.shape[0] == 0 or kpts_B_sph.shape[0] == 0:
        return {'matches': np.empty((0, 2), dtype=int),
                'gt_matches0': np.full(kpts_A_sph.shape[0], -1, dtype=int),
                'gt_matches1': np.full(kpts_B_sph.shape[0], -1, dtype=int)}

    W, H = 2048, 1024 # Standard for Spherecraft training data

    kptsA = standard_spherical_to_pixel(kpts_A_sph, W, H)
    pose = (pose_B_w2c.dot(pose_A_c2w)).astype(float)
    params = {'pose01': pose, 'depth0': depth_map_A,
              'depth1': depth_map_B, 'width': W, 'height': H,
              'abs_tol': 0.05, 'rel_tol': 0.02}

    k0_final, k1_final, final_ids, _ = warp_se3_spherical(kptsA, params)

    if len(final_ids) == 0:
        # No matches found
        gt_matches0 = np.full(kpts_A_sph.shape[0], -1, dtype=int)
        gt_matches1 = np.full(kpts_B_sph.shape[0], -1, dtype=int)
        matches_pairs = np.empty((0, 2), dtype=int)
    else:
        # Convert k1_final to k1_final_sph and then to k1_final_xyz
        k1_final_sph = standard_pixel_to_spherical(k1_final, W, H)
        k1_final_xyz = spherical_to_cartesian(k1_final_sph)

        # Convert kpts_B_sph to kpts_B_xyz
        kpts_B_xyz = spherical_to_cartesian(kpts_B_sph)
        
        # Calculate similarity and find nearest neighbors         
        similarity_matrix = np.einsum('ik,jk->ij', k1_final_xyz, kpts_B_xyz)
        similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
        angular_distance_matrix = np.rad2deg(np.arccos(similarity_matrix))

        best_match_for_A = np.argmin(angular_distance_matrix, axis=1)
        min_distances_for_A = np.min(angular_distance_matrix, axis=1)
        best_match_for_B = np.argmin(angular_distance_matrix, axis=0)
        
        
        mutual_mask = (best_match_for_B[best_match_for_A] == np.arange(len(k1_final_xyz)))
        distance_mask = min_distances_for_A < angle_threshold_degrees
        final_mask = mutual_mask & distance_mask

        # Get the valid projected A indices (these correspond to final_ids)
        valid_projected_indices = np.arange(len(k1_final_xyz))[final_mask]

        matched_indices_A = final_ids[valid_projected_indices]
        matched_indices_B = best_match_for_A[final_mask]

        # 5. Format output
        gt_matches0 = np.full(kpts_A_sph.shape[0], -1, dtype=int)
        gt_matches1 = np.full(kpts_B_sph.shape[0], -1, dtype=int)
        
        gt_matches0[matched_indices_A] = matched_indices_B
        gt_matches1[matched_indices_B] = matched_indices_A
        matches_pairs = np.stack([matched_indices_A, matched_indices_B], axis=1)
        # print("\n\n --- No. of Matches ---\n",len(matches_pairs))

    return {'matches': matches_pairs, 'gt_matches0': gt_matches0, 'gt_matches1': gt_matches1}


# ==============================================================================
# PIPELINE STAGES
# ==============================================================================

def run_feature_extraction(image_dir: Path, feature_dir: Path, num_keypoints: int):
    """
    Stage 1: Extracts spherical XFeat keypoints for all images.
    """
    logging.info("--- Stage 1: Running Feature Extraction ---")
    image_files = [p for p in image_dir.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    feature_dir.mkdir(exist_ok=True, parents=True)
    
    def worker_extract(img_path: Path):
        try:
            output_path = feature_dir / f"{img_path.stem}.npz"
            if output_path.exists():
                return f"Skipped {img_path.name}, features already exist."

            image_equi = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image_equi is None:
                return f"Error loading {img_path.name}"

            # Project to dicemap for CNN-based feature extractor
            image_dicemap = equirectangular_to_dicemap(image_equi)

            # Generate keypoints (this util converts kpts from dicemap back to spherical)
            kpts, descs, scores = generate_keypoints(image_dicemap, num_keypoints)
            
            np.savez_compressed(
                output_path,
                keypoints=kpts, # Shape (N, 2) in (phi, theta)
                descriptors=descs,
                scores=scores,
            )
            return f"Processed {img_path.name}"
        except Exception as e:
            return f"Failed {img_path.name}: {e}"

    results = Parallel(n_jobs=8, verbose=1)(delayed(worker_extract)(p) for p in image_files)
    for res in results:
        if "Error" in res or "Failed" in res:
            logging.warning(res)
        else:
            logging.info(res)


def generate_finetuning_pairs(config: Dict):
    """
    Stage 2: Generates the final .npz pair files using the assets.
    """
    logging.info("--- Stage 2: Generating Finetuning Pairs ---")
    
    # 1. Load the list of pairs to process
    pairs_to_process = read_pairs_from_txt(config['pairs_path'])
    
    # 2. Create output directory
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    
    # 3. Process pairs in parallel
    def worker_process_pair(stem1: str, stem2: str):
        # try:
        output_filename = f"{config['dataset_name']}_{stem1}_{stem2}.npz"
        output_path = config['output_dir'] / output_filename
        # if output_path.exists():
        #     return f"Skipped pair {stem1}-{stem2}, already exists."

        # Load features for both images
        features1 = np.load(config['feature_dir'] / f"{stem1}.npz")
        features2 = np.load(config['feature_dir'] / f"{stem2}.npz")
        
        # Load depth map for the first image
        depth_path1 = config['depth_dir'] / f"{stem1}.exr"
        depth_path2 = config['depth_dir'] / f"{stem2}.exr"
        if not (depth_path1.exists() and depth_path2.exists()):
            return f"Warning: Depth map not found for {stem1} or {stem2}, skipping pair."
        
        
        try:
            depth_map1 = read_depth(str(depth_path1))
            depth_map2 = read_depth(str(depth_path2))
        except Exception as e:
            return f"depth map is the problem. {e}"

        # Load poses for both images
        try:
            p1_c2w, p1_w2c = get_c2w_and_w2c_matrix(config['pose_dir'] / f"{stem1}.dat")
            p2_c2w, p2_w2c = get_c2w_and_w2c_matrix(config['pose_dir'] / f"{stem2}.dat")
        except Exception as e:
            return f"Pose is the problem. {e}"

        # Generate ground truth by projecting points from 1 to 2
        gt_data = generate_sfm_groundtruth(
            kpts_A_sph=features1['keypoints'],
            kpts_B_sph=features2['keypoints'],
            depth_map_A=depth_map1,
            depth_map_B=depth_map2,
            pose_A_c2w=p1_c2w,
            pose_B_w2c=p2_w2c,
            angle_threshold_degrees=config['angle_threshold']
        )

        # Save the final npz file in a LightGlue-compatible format
        # image_size for spherical data is constant: 2*pi radians for width (theta), pi for height (phi)
        spherical_image_size = torch.tensor([2 * np.pi, np.pi]) 
        temp_path = output_path.with_suffix('.npz.tmp')
        np.savez(
            temp_path,
            keypoints0=features1['keypoints'],
            descriptors0=features1['descriptors'],
            scores0=features1['scores'],
            image_size0=spherical_image_size,
            
            keypoints1=features2['keypoints'],
            descriptors1=features2['descriptors'],
            scores1=features2['scores'],
            image_size1=spherical_image_size,

            matches=torch.from_numpy(gt_data['matches']).long(),
            gt_matches0=torch.from_numpy(gt_data['gt_matches0']).long(),
            gt_matches1=torch.from_numpy(gt_data['gt_matches1']).long(),
        )
        temp_path.rename(output_path)
        
        return f"Saved pair {output_filename}"

        # except Exception as e:
        #     return f"Failed to process pair {stem1}-{stem2}: {e}"

    logging.info(f"Starting to process {len(pairs_to_process)} pairs in parallel...")
    results = Parallel(n_jobs=20, verbose=1)(delayed(worker_process_pair)(p[0], p[1]) for p in pairs_to_process)
    for res in results:
        if "Warning" in res or "Failed" in res:
            logging.warning(res)
        else:
            logging.info(res)

# ==============================================================================
# MAIN ORCHESTRATION
# ==============================================================================

def main():
    # --- Configuration for your 'barbershop' dataset ---
    DATASET_NAME = "berlin"
    BASE_PATH = Path(f"/data/code/glue-factory/datasets/spherecraft_data/{DATASET_NAME}")
    OUTPUT_PATH = Path("/data/code/glue-factory/data/finetuning/finetuning_pairs_spherecraft")
    
    CONFIG = {
        # Input Paths
        "pairs_path": BASE_PATH / "superpoint_train.txt",
        "image_dir": BASE_PATH / "images",
        "depth_dir": BASE_PATH / "depthmaps",
        "pose_dir": BASE_PATH / "extr",

        # Output Paths
        "feature_dir": BASE_PATH / "features_xfeat_spherical",
        "output_dir": OUTPUT_PATH,

        # Parameters
        "num_keypoints": 2048,
        "angle_threshold": 1, # Angular threshold in degrees for a match

        # Dataset Name
        "dataset_name": DATASET_NAME
    }

    # --- Execute Pipeline ---

    # 1. Feature Extraction (can be skipped if already done)
    # Set to True to skip this step
    SKIP_FEATURE_EXTRACTION = True
    if not SKIP_FEATURE_EXTRACTION:
        run_feature_extraction(CONFIG['image_dir'], CONFIG['feature_dir'], CONFIG['num_keypoints'])
    else:
        logging.info("Skipping feature extraction as requested.")
        
        
    # 2. Pair Generation
    generate_finetuning_pairs(CONFIG)
    
    logging.info("--- Pipeline Finished ---")
    logging.info(f"Output saved to: {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()