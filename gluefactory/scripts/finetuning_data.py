import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any
import OpenEXR as exr
import Imath
import numpy as np
import torch
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from joblib import Parallel, delayed
import argparse

# --- Metashape Import ---
try:
    import Metashape
except ImportError:
    print("Warning: Metashape module not found. Some functions will not be available.")
    Metashape = None

# --- Utils ---
from gluefactory.utils.covisibility_graph import NVMParser
from gluefactory.settings import DATA_PATH # Assuming this is where datasets are stored
from gluefactory.utils.image import ImagePreprocessor, load_image # Use Glue Factory's image utils
from gluefactory.utils.equirectangular_utils import equirectangular_to_dicemap # Import equirectangular utils
from gluefactory.utils.spherical_utils import standard_spherical_to_pixel, cartesian_to_spherical, spherical_to_cartesian # Import spherical utils
from gluefactory.utils.xfeat_utils import generate_keypoints # Import xfeat utils
from gluefactory.datasets.base_dataset import BaseDataset # Crucial import


def parse_nvm_cameras(nvm_parser: NVMParser, model_index: int = 0) -> Dict[str, Dict[str, Any]]:
    """
    Parses camera data from NVM into a more usable format with poses.
    
    Returns:
        A dictionary mapping filename to its pose data (R, C, f).
    """
    camera_data = {}
    model = nvm_parser.models[model_index]

    for cam in model['cameras']:
        filename = Path(cam['filename']).name
        data = cam['data']
        
        # <f> <qw> <qx> <qy> <qz> <cx> <cy> <cz> <radial_distortion> 0
        focal_length = float(data[0])
        q = np.array([float(d) for d in data[1:5]]) # qw, qx, qy, qz
        center = np.array([float(d) for d in data[5:8]]) # Camera center C

        # Convert quaternion to rotation matrix R
        # Note: NVM quaternion is (w, x, y, z) -- http://ccwu.me/vsfm/doc.html#nvm
        w, x, y, z = q
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        
        # The NVM stores C = -R't, so the translation vector t = -RC
        t = -R @ center
        
        camera_data[filename] = {"R": R, "t": t, "C": center, "f": focal_length}
    return camera_data


def get_depth_at_spherical_coords(depth_map: np.ndarray, kpts_spherical: np.ndarray) -> np.ndarray:
    """Samples depth values from an equirectangular depth map for given spherical coords."""
    h, w = depth_map.shape[:2]
    
    # Convert spherical to pixel coordinates
    # (u,v) -> (x,y)
    # 1. Call the function and store the result in a single variable
    pixel_coords = standard_spherical_to_pixel(kpts_spherical, w, h)
    
    # 2. Unpack the columns from the resulting array
    px = pixel_coords[:, 0]
    py = pixel_coords[:, 1]
    
    # 3. Convert to integer for indexing and clamp to valid range
    px = np.clip(px, 0, w - 1).astype(int)
    py = np.clip(py, 0, h - 1).astype(int)

    # 4. Use the integer coordinates to sample the depth map
    return depth_map[py, px]


def read_EXR(filepath: str, channel_name: str = 'R'):
    """
    Reads an EXR file and converts a specified channel into a NumPy array.

    Args:
        filepath (str): The path to the EXR file.
        channel_name (str): The name of the channel to extract (e.g., 'R', 'G', 'B', 'A', 'Z', 'unknown 0').

    Returns:
        numpy.ndarray: The extracted channel data as a NumPy array.
    """
    exrfile = exr.InputFile(filepath)
    header = exrfile.header()
    
    # Determine the data window for correct image dimensions
    data_window = header['dataWindow']
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1

    # Get the pixel type of the channel (e.g., FLOAT)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

    # Read the raw bytes of the specified channel
    raw_bytes = exrfile.channel(channel_name, pixel_type)

    # Convert raw bytes to a NumPy array and reshape to image dimensions
    data_array = np.frombuffer(raw_bytes, dtype=np.float32)
    data_array = np.reshape(data_array, (height, width))

    return data_array


# ==============================================================================
# PIPELINE STAGES
# ==============================================================================

def run_feature_extraction(image_dir: Path, feature_dir: Path, num_keypoints: int):
    """
    Stage 1: Extracts and saves keypoints for all images in parallel.
    """
    logging.info("--- Stage 1: Running Feature Extraction ---")
    image_files = os.listdir(image_dir)
    feature_dir.mkdir(exist_ok=True)
    
    def worker_extract(img):
        try:
            output_path = feature_dir / f"{img}.npz"
            if output_path.exists():
                return f"Skipped {img}, features already exist."

            image_equi = cv2.imread(f"{image_dir}/"+str(img), 1)
            if image_equi is None:
                return f"Error loading {img}"

            image_dicemap = equirectangular_to_dicemap(image_equi)

            # internally convert dicemap keypoints back to equirectangular keypoints
            kpts, descs, scores = generate_keypoints(image_dicemap, num_keypoints)
            
            np.savez_compressed(
                output_path,
                keypoints=kpts, # Shape (N, 2) in (phi, theta)
                descriptors=descs,
                scores=scores,
                image_size=np.array(image_equi.shape[:2]) # H, W
            )
            return f"Processed {img}"
        except Exception as e:
            return f"Failed {img}: {e}"

    results = Parallel(n_jobs=4, verbose=10)(delayed(worker_extract)(p) for p in image_files)
    for res in results:
        if "Error" in res or "Failed" in res:
            logging.warning(res)
        else:
            logging.info(res)


def generate_sfm_groundtruth(
    kpts_A_sph: np.ndarray,
    kpts_B_sph: np.ndarray,
    depth_map_A: np.ndarray,
    pose_A: Dict[str, np.ndarray],
    pose_B: Dict[str, np.ndarray],
    angle_threshold_degrees: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Generates ground truth matches using SfM poses and a depth map.
    This replaces the rotation-based ground truth function.
    """
    if kpts_A_sph.shape[0] == 0 or kpts_B_sph.shape[0] == 0:
        return {'matches': np.empty((0, 2), dtype=int),
                'gt_matches0': np.full(kpts_A_sph.shape[0], -1, dtype=int),
                'gt_matches1': np.full(kpts_B_sph.shape[0], -1, dtype=int)}

    # 1. Get relative pose from A to B
    # P_B = R_B @ (R_A.T @ (P_A - C_A)) + C_B is not quite right.
    # World point P_w = R_A.T @ P_A_cam + C_A
    # Point in B's frame P_B_cam = R_B @ (P_w - C_B)
    # P_B_cam = R_B @ (R_A.T @ P_A_cam + C_A - C_B)
    R_A, C_A = pose_A['R'], pose_A['C']
    R_B, C_B = pose_B['R'], pose_B['C']
    R_AB = R_B @ R_A.T
    t_AB = R_B @ (C_A - C_B)

    # 2. Get depth for all keypoints in A
    depths_A = get_depth_at_spherical_coords(depth_map_A, kpts_A_sph)
    valid_depth_mask = depths_A > 0

    if not np.any(valid_depth_mask):
        return {'matches': np.empty((0, 2), dtype=int),
                'gt_matches0': np.full(kpts_A_sph.shape[0], -1, dtype=int),
                'gt_matches1': np.full(kpts_B_sph.shape[0], -1, dtype=int)}

    # 3. Project valid keypoints from A to 3D, then to B's frame
    kpts_A_valid_sph = kpts_A_sph[valid_depth_mask]
    kpts_A_valid_3d_local = spherical_to_cartesian(kpts_A_valid_sph) * depths_A[valid_depth_mask, np.newaxis]
    
    # Transform to B's coordinate system
    projected_kpts_in_B_3d = (R_AB @ kpts_A_valid_3d_local.T).T + t_AB
    projected_kpts_in_B_sph = cartesian_to_spherical(projected_kpts_in_B_3d)

    # 4. Find matches based on angular distance
    # Convert to unit vectors for angular distance calculation
    projected_kpts_in_B_xyz = spherical_to_cartesian(projected_kpts_in_B_sph)
    kpts_B_xyz = spherical_to_cartesian(kpts_B_sph)
    
    similarity_matrix = np.einsum('ik,jk->ij', projected_kpts_in_B_xyz, kpts_B_xyz)
    similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
    angular_distance_matrix = np.rad2deg(np.arccos(similarity_matrix))

    # 5. Mutual Nearest Neighbor check
    best_match_for_A = np.argmin(angular_distance_matrix, axis=1)
    min_distances_for_A = np.min(angular_distance_matrix, axis=1)
    best_match_for_B = np.argmin(angular_distance_matrix, axis=0)
    
    # The indices of the valid keypoints from the original kpts_A list
    original_indices_A = np.where(valid_depth_mask)[0]
    
    # Check for mutual consistency
    mutual_mask = (best_match_for_B[best_match_for_A] == np.arange(len(original_indices_A)))
    distance_mask = min_distances_for_A < angle_threshold_degrees
    final_mask = mutual_mask & distance_mask

    # Get the indices from the *original* lists
    matched_indices_A = original_indices_A[final_mask]
    matched_indices_B = best_match_for_A[final_mask]

    # 6. Format output
    gt_matches0 = np.full(kpts_A_sph.shape[0], -1, dtype=int)
    gt_matches1 = np.full(kpts_B_sph.shape[0], -1, dtype=int)
    
    gt_matches0[matched_indices_A] = matched_indices_B
    gt_matches1[matched_indices_B] = matched_indices_A
    matches_pairs = np.stack([matched_indices_A, matched_indices_B], axis=1)

    return {'matches': matches_pairs, 'gt_matches0': gt_matches0, 'gt_matches1': gt_matches1}



def generate_finetuning_pairs(config: Dict):
    """
    Stage 2: Generates the final .npz pair files using pre-computed assets.
    """
    logging.info("--- Stage 2: Generating Finetuning Pairs ---")
    
    # 1. Load all necessary pre-computed data
    logging.info("Loading NVM file and building covisibility graph...")
    parser = NVMParser()
    parser.parse(str(config['nvm_path']))
    graph = parser.build_covisibility_graph()
    camera_poses = parse_nvm_cameras(parser)
    
    # Map from camera index to filename
    cam_idx_to_name = parser.camera_map[0]

    # 2. Bin pairs by covisibility score
    logging.info("Binning pairs by covisibility score...")
    all_pairs = []
    for i, neighbors in graph.items():
        for j, score in neighbors.items():
            if i < j:
                all_pairs.append({'cam_i': i, 'cam_j': j, 'score': score})

    scores = [p['score'] for p in all_pairs]
    if not scores:
        logging.error("No pairs found in covisibility graph. Exiting.")
        return
        
    p_medium, p_hard = np.quantile(scores, config['bin_quantiles'])
    bins = {'easy': [], 'medium': [], 'hard': []}
    for pair in all_pairs:
        if pair['score'] > p_medium:
            bins['easy'].append(pair)
        elif pair['score'] > p_hard:
            bins['medium'].append(pair)
        else:
            bins['hard'].append(pair)

    logging.info(f"Found {len(bins['easy'])} easy, {len(bins['medium'])} medium, {len(bins['hard'])} hard pairs.")

    # 3. Create output directories
    for bin_name in bins.keys():
        (config['output_dir'] / bin_name).mkdir(parents=True, exist_ok=True)
    
    # 4. Process pairs in parallel
    def worker_process_pair(pair_info, bin_name):
        try:
            cam_i_idx, cam_j_idx = pair_info['cam_i'], pair_info['cam_j']
            
            cam_i_name = Path(cam_idx_to_name[cam_i_idx]).name
            cam_j_name = Path(cam_idx_to_name[cam_j_idx]).name
            
            output_filename = f"{Path(cam_i_name).stem}_{Path(cam_j_name).stem}.npz"
            output_path = config['output_dir'] / bin_name / output_filename
            if output_path.exists():
                return f"Skipped pair {cam_i_name}-{cam_j_name}, already exists."

            # Load features and depth 
            features_i = np.load(config['feature_dir'] / f"{Path(cam_i_name).stem}.npz")
            features_j = np.load(config['feature_dir'] / f"{Path(cam_j_name).stem}.npz")
            depth_i_path = config['depth_dir'] / f"{Path(cam_i_name).stem}.exr"
            
            if not depth_i_path.exists():
                return f"Warning: Depth map not found for {cam_i_name}, skipping pair."
            
            try:
                depth_map_i = read_EXR(str(depth_i_path), channel_name='R')
            except Exception:
                depth_map_i = read_EXR(str(depth_i_path), channel_name='unknown 0')

            if depth_map_i.ndim == 3: # Handle multi-channel depth maps if necessary
                depth_map_i = depth_map_i[:, :, 0]

            # Generate ground truth
            try:
                gt_data = generate_sfm_groundtruth(
                    kpts_A_sph=features_i['keypoints'],
                    kpts_B_sph=features_j['keypoints'],
                    depth_map_A=depth_map_i,
                    pose_A=camera_poses[cam_i_name],
                    pose_B=camera_poses[cam_j_name],
                    angle_threshold_degrees=config['angle_threshold']
                )
            except Exception as e:
                return f"Failed to generate ground truth {pair_info}: {e}"

            # Save the final npz file in the LightGlue-compatible format
            # Keypoints are (phi, theta), image_size is for the equirectangular domain
            equi_image_size = torch.tensor([2 * np.pi, np.pi]) 

            np.savez(
                output_path,
                keypoints0=features_i['keypoints'],
                descriptors0=features_i['descriptors'],
                scores0=features_i['scores'],
                image_size0=equi_image_size,
                
                keypoints1=features_j['keypoints'],
                descriptors1=features_j['descriptors'],
                scores1=features_j['scores'],
                image_size1=equi_image_size,

                matches=torch.from_numpy(gt_data['matches']).long(),
                gt_matches0=torch.from_numpy(gt_data['gt_matches0']).long(),
                gt_matches1=torch.from_numpy(gt_data['gt_matches1']).long(),
            )
            return f"Saved pair {output_filename} to {bin_name}."

        except Exception as e:
            return f"Failed to process pair {pair_info}: {e}"

    all_tasks = []
    for bin_name, pair_list in bins.items():
        if bin_name in ['easy', 'medium']:
            for pair in pair_list:
                all_tasks.append((pair, bin_name))
            
    logging.info(f"Starting to process {len(all_tasks)} pairs in parallel...")
    results = Parallel(n_jobs=4, verbose=10)(delayed(worker_process_pair)(task[0], task[1]) for task in all_tasks)
    for res in results:
        if "Warning" in res or "Failed" in res:
            logging.warning(res)
        else:
            logging.info(res)

# ==============================================================================
# MAIN ORCHESTRATION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate LightGlue fine-tuning data from Metashape projects.")
    parser.add_argument("project_path", type=Path, help="Path to the job id")
    parser.add_argument("output_dir", type=Path, help="Path to the root directory where all outputs will be stored.")
    parser.add_argument("--skip_feature_extraction", action="store_true", help="Skip feature extraction if already done.")
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # --- Configuration ---
    CONFIG = {
        # Paths
        "psz_path": args.project_path / "pointcloud/project.psz",
        "image_dir": args.project_path / "images",
        "output_dir": args.output_dir,
        "depth_dir": args.project_path / "depth",
        "feature_dir": args.project_path / "features",
        "nvm_path": args.project_path / "pointcloud/cameras.nvm",

        # Parameters
        "num_keypoints": 2048,
        "bin_quantiles": [0.66, 0.33], # Quantiles for easy/medium/hard split
        "angle_threshold": 0.3, # Angular threshold in degrees for a match
    }

    # Create output directories
    for dir_path in [CONFIG['output_dir'], CONFIG['depth_dir'], CONFIG['feature_dir']]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # --- Execute Pipeline ---
    
    # 0. Generate assets from Metashape (Depth + Cameras)
    # This part requires the Metashape library to be active.
    if Metashape:
        logging.info("--- Initial Asset Generation from Metashape ---")
        doc = Metashape.Document()
        doc.open(str(CONFIG['psz_path']))
        chunk = doc.chunk
        
        # Export cameras if NVM doesn't exist
        if not CONFIG['nvm_path'].exists():
            logging.info(f"Exporting cameras to {CONFIG['nvm_path']}...")
            chunk.exportCameras(path=str(CONFIG['nvm_path']))
        else:
            logging.info("NVM file already exists, skipping export.")
            
        # Export depth maps
        logging.info("Generating and saving depth maps...")
        if len(os.listdir(CONFIG['depth_dir'])) != len(os.listdir(CONFIG['image_dir'])):
            for camera in chunk.cameras:
                depth_path = CONFIG['depth_dir'] / f"{camera.label}.exr"
                if not depth_path.exists():
                    depth = chunk.model.renderDepth(camera.transform, camera.sensor.calibration)
                    depth.convert("R", "F16")
                    depth.save(str(depth_path))
                    logging.info(f"Saved depth for {camera.label}")
            logging.info("Depth map generation complete.")
        else:
            logging.info("Depth maps already exist, skipping generation.")

    else:
        logging.warning("Metashape not found. Assuming depth maps and NVM file already exist.")
        if not CONFIG['nvm_path'].exists():
            logging.error(f"Metashape not found and NVM file is missing at {CONFIG['nvm_path']}. Cannot proceed.")
            sys.exit(1)


    # 1. Feature Extraction
    if not args.skip_feature_extraction:
        run_feature_extraction(CONFIG['image_dir'], CONFIG['feature_dir'], CONFIG['num_keypoints'])
    else:
        logging.info("Skipping feature extraction as requested.")
        
    # 2. Pair Generation
    generate_finetuning_pairs(CONFIG)
    
    logging.info("--- Pipeline Finished ---")

if __name__ == "__main__":
    main()