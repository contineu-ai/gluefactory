import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset # Direct import is fine
import cv2 # For image loading, if not using gluefactory.utils.image.load_image

from ..settings import DATA_PATH # Assuming this is where datasets are stored
from ..utils.image import ImagePreprocessor, load_image # Use Glue Factory's image utils
from .base_dataset import BaseDataset # Crucial import

logger = logging.getLogger(__name__)

class SphereCraftDataset(BaseDataset):
    default_conf = {
        "batch_size": 1,
        "num_workers": 2,
        "prefetch_factor": 2,
        "train_batch_size": 2,
        "val_batch_size": None,
        "test_batch_size": None,

        # --- Add your CUSTOM keys here ---
        "data_root": "data/spherecraft_data", # Default path
        "train_list_suffix": "train.txt",
        "val_list_suffix": "val.txt",
        "test_list_suffix": "test.txt", # Even if not used now, good to have
        "image_native_width": 1024,
        "image_native_height": 512,
        # --- Paths ---
        "data_dir": "spherecraft_data/", # Root for all SphereCraft scenes, relative to DATA_PATH
        "scene_name": "seoul", # Specific scene to load, e.g., 'seoul', 'shapespark' (MANDATORY)
        "image_subdir": "images",
        "keypoints_detector": "superpoint", # e.g., 'superpoint', 'sift'

        # --- File lists (relative to scene_name directory) ---
        # These

        # --- Image Options ---
        "read_image": True,
        "grayscale": True, # LightGlue typically uses grayscale
        "image_native_width": 2048, # Native width of SphereCraft images for conversion
        "image_native_height": 1024, # Native height for conversion
        "preprocessing": ImagePreprocessor.default_conf, # For resizing, normalization

        # --- Misc ---
        # "views": 2, # Implicitly always 2 for this dataset intended for pair matching
        # "reseed": False, # from BaseDataset's worker_init_fn
        # "seed": 0, # from BaseDataset
        # "num_threads": 1, # from BaseDataset
    }

    def _init(self, conf):
        """Initialization method, called by BaseDataset."""
        self.conf = conf # self.conf is already set by BaseDataset __init__

        self.scene_root = DATA_PATH / self.conf.data_dir / self.conf.scene_name
        if not self.scene_root.exists():
            raise FileNotFoundError(
                f"Scene '{self.conf.scene_name}' not found at {self.scene_root}. "
                "Please download it first using the SphereCraft download script."
            )

        self.kpt_detector_name = self.conf.keypoints_detector
        self.keypoints_dir = self.scene_root / "keypoints" / f"spherical_{self.kpt_detector_name}"
        # SphereCraft's GT dir name includes the detector and "_gt_matches"
        self.gt_corr_dir = self.scene_root / "gt_correspondences" / f"{self.kpt_detector_name}_gt_matches"
        self.image_dir = self.scene_root / self.conf.image_subdir

        if not self.keypoints_dir.exists():
            raise FileNotFoundError(f"Keypoints directory not found: {self.keypoints_dir}")
        if not self.gt_corr_dir.exists():
            raise FileNotFoundError(f"Ground truth correspondences directory not found: {self.gt_corr_dir}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # The actual torch.utils.data.Dataset instances will be created in get_dataset
        logger.info(f"Initialized SphereCraftDataset for scene: {self.conf.scene_name} with detector: {self.kpt_detector_name}")


    def get_dataset(self, split):
        """Returns an instance of torch.utils.data.Dataset for the
            requested split ('train', 'val', or 'test')."""
        assert split in ["train", "val", "test"], f"Unknown split: {split}"
        return _PairDatasetSphereCraft(self.conf, split, self.scene_root, self.image_dir, self.keypoints_dir, self.gt_corr_dir)


class _PairDatasetSphereCraft(Dataset): # Standard PyTorch Dataset
    def __init__(self, conf, split, scene_root, image_dir, keypoints_dir, gt_corr_dir):
        self.conf = conf
        self.split = split
        self.scene_root = scene_root
        self.image_dir = image_dir
        self.keypoints_dir = keypoints_dir
        self.gt_corr_dir = gt_corr_dir

        self.preprocessor = ImagePreprocessor(self.conf.preprocessing)

        # Determine the list file to load
        list_suffix_key = f"{self.split}_list_suffix"
        if list_suffix_key not in self.conf:
            raise ValueError(f"Configuration for '{list_suffix_key}' is missing for split '{self.split}'.")
        list_filename = f"{self.conf.keypoints_detector}_{self.conf[list_suffix_key]}"
        list_file_path = self.scene_root / list_filename

        if not list_file_path.exists():
            raise FileNotFoundError(f"List file not found for split '{self.split}': {list_file_path}")

        self.items = []
        with open(list_file_path, 'r') as f:
            for line_num, line in enumerate(f):
                pair = line.strip().split("/")[-1].split(".")[0].split("_") 
                if len(pair) == 2:
                    # Store relative image names (e.g., "00000000.jpg")
                    self.items.append((f"{pair[0]}.jpg", f"{pair[1]}.jpg"))
                else:
                    logger.warning(f"Skipping malformed line {line_num+1} in {list_file_path}: '{line.strip()}'")
        
        if not self.items:
            logger.warning(f"No items loaded for split '{self.split}' from {list_file_path}.")

        logger.info(f"Loaded {len(self.items)} pairs for split '{self.split}' from {list_file_path}.")


    def _read_view_data(self, image_filename_jpg):
        """Helper to load and preprocess data for a single view."""
        base_name = Path(image_filename_jpg).stem

        # --- Load Image ---
        image_path_full = self.image_dir / image_filename_jpg
        if not image_path_full.exists():
            raise FileNotFoundError(f"Image file not found: {image_path_full}")
        # load_image returns a NumPy array (H, W, C) or (H, W)
        img_raw_np = load_image(image_path_full, self.conf.grayscale)

        # --- Load Keypoints & Descriptors ---
        kpt_file_path = self.keypoints_dir / f"seoul_img_{base_name}.npz"
        if not kpt_file_path.exists():
            raise FileNotFoundError(f"Keypoint file not found: {kpt_file_path}")
        kpt_data = np.load(kpt_file_path)

        kpts_spherical_np = kpt_data['keypointCoords'].reshape(-1, 2)  
        descriptors_np = kpt_data['keypointDescriptors'].reshape(-1, 256)  
        scores_np = kpt_data['keypointScores']

        keypoints_px_native_np = self.spherical_to_pixel(
            kpts_spherical_np,
            self.conf.image_native_width,
            self.conf.image_native_height
        )

        # --- Preprocessing (Resizing, Normalization) ---
        # ImagePreprocessor expects HWC format for color, HW for grayscale
        # load_image provides this.
        proc_out = self.preprocessor(img_raw_np)
        image_processed_torch = proc_out['image'] # Tensor, CHW
        
        # scales are (scale_w, scale_h) by which original dims were multiplied
        scales_wh_np = proc_out['scales']

        # Scale keypoints to the processed image dimensions
        keypoints_scaled_px_np = keypoints_px_native_np * scales_wh_np.cpu().numpy()
        
        return {
            'name': image_filename_jpg,
            'image': image_processed_torch,
            'keypoints': torch.from_numpy(keypoints_scaled_px_np).float(),
            'descriptors': torch.from_numpy(descriptors_np).float(),
            'scores': torch.from_numpy(scores_np).float(),
            'image_size': torch.tensor(image_processed_torch.shape[-2:][::-1]).float(), # W, H of processed image
        }

    def spherical_to_pixel(self, kpts_sph_np, W_native, H_native):
        Pi = np.pi
        TwoPi = 2 * np.pi

        phi = kpts_sph_np[:, 0]
        theta = kpts_sph_np[:, 1]

        # 1. Normalize phi and theta to be within [0, 2*pi)
        phi = phi % TwoPi
        
        # 2. Handle hemisphere reflection for phi
        # Find points in the "southern hemisphere" (phi >= pi)
        southern_mask = phi >= Pi
        # Reflect their phi to the northern hemisphere
        phi[southern_mask] = TwoPi - phi[southern_mask]
        # And rotate their theta by 180 degrees
        theta[southern_mask] += Pi

        # 3. Normalize the adjusted theta to be within [0, 2*pi)
        theta = theta % TwoPi
    
        # Calculate pixel x coordinate (note the "1.0 - ..." for correct direction)
        x_px = W_native * (1.0 - (theta / TwoPi)) - 0.5
        
        # Calculate pixel y coordinate
        y_px = H_native * (phi / Pi) - 0.5
        
        # The original code clipped the values, which is good practice.
        x_px = np.clip(x_px, 0, W_native - 1)
        y_px = np.clip(y_px, 0, H_native - 1)

        return np.stack([x_px, y_px], axis=-1)
    
    def __getitem__(self, idx):
        image_name0_jpg, image_name1_jpg = self.items[idx]

        # The worker_init_fn in BaseDataset handles seeding if self.conf.reseed is true
        # So, no need for explicit fork_rng here unless more complex per-item seeding is needed.

        data0 = self._read_view_data(image_name0_jpg)
        data1 = self._read_view_data(image_name1_jpg)

        # --- Load Ground Truth Correspondences ---
        base_name0 = Path(image_name0_jpg).stem
        base_name1 = Path(image_name1_jpg).stem

        gt_match_filename = f"{base_name0}_{base_name1}.npz"
        gt_match_file_path = self.gt_corr_dir / gt_match_filename
        
        # SphereCraft GT files are typically from image0 to image1 if N0 < N1, etc.
        # For simplicity, we assume the file exists as named. Robust code might check reverse.
        
        if not gt_match_file_path.exists():
            # This is an issue for training if no GT is found.
            # For evaluation, it might mean no GT overlap.
            logger.error(f"GT match file not found: {gt_match_file_path} or its reverse for pair ({base_name0}, {base_name1}).")
            # Return dummy matches or raise error if in 'train' split
            if self.split == 'train':
                raise FileNotFoundError(f"Required GT match file for training not found: {gt_match_file_path}")
            # For val/test, we might proceed with no GT matches
            gt_matches_indices_np = np.full(data0['keypoints'].shape[0], -1, dtype=np.int64)

        if gt_match_file_path.exists(): # Check again in case it was found (original or reversed)
            gt_data = np.load(gt_match_file_path)
            gt_matches_indices_np = gt_data['correspondences'] # Indices from perspective of file's first image

        num_kpts0 = data0['keypoints'].shape[0]
        num_kpts1 = data1['keypoints'].shape[0]
        
        # Ensure correspondence array matches the number of keypoints in the source image
        if len(gt_matches_indices_np) != num_kpts0:
            logger.warning(
                f"Mismatch: num kpts in {base_name0} ({num_kpts0}) vs "
                f"GT correspondence array_len ({len(gt_matches_indices_np)}) for pair ({base_name0},{base_name1}). Truncating/padding GT."
            )
            # Pad with -1 or truncate. Prefer padding if gt is shorter, truncate if gt is longer.
            if len(gt_matches_indices_np) < num_kpts0:
                padded_gt = np.full(num_kpts0, -1, dtype=np.int64)
                padded_gt[:len(gt_matches_indices_np)] = gt_matches_indices_np
                gt_matches_indices_np = padded_gt
            else: # gt is longer
                gt_matches_indices_np = gt_matches_indices_np[:num_kpts0]


        gt_matches0 = torch.from_numpy(gt_matches_indices_np).long()
        valid0 = (gt_matches0 != -1)

        # gt_matches1 - (the inverse mapping from kpts1 to kpts0)
        gt_matches1 = torch.full((num_kpts1,), -1, dtype=torch.long)
        # Get the indices of keypoints in image 0 that have a valid match
        matched_indices_in_0 = torch.where(valid0)[0]
        # Get the corresponding indices in image 1
        corresponding_indices_in_1 = gt_matches0[matched_indices_in_0]

        # Ensure indices are within the bounds of kpts1 before assigning
        if corresponding_indices_in_1.numel() > 0:
            valid_mask_1 = corresponding_indices_in_1 < num_kpts1
            matched_indices_in_0 = matched_indices_in_0[valid_mask_1]
            corresponding_indices_in_1 = corresponding_indices_in_1[valid_mask_1]
            
            # Populate the reverse map
            gt_matches1[corresponding_indices_in_1] = matched_indices_in_0


        return {
            'image0': data0['image'],
            'keypoints0': data0['keypoints'],
            'descriptors0': data0['descriptors'],
            'scores0': data0['scores'],
            'image_size0': data0['image_size'], # W, H of processed image0

            'image1': data1['image'],
            'keypoints1': data1['keypoints'],
            'descriptors1': data1['descriptors'],
            'scores1': data1['scores'],
            'image_size1': data1['image_size'], # W, H of processed image1

            # These are the keys the loss function needs
            'matches0': gt_matches0,  # [M,]: for each kpt in image 0, the matched kpt index in image 1 (-1 if no match)
            'gt_matches0': gt_matches0,       # [M,]: for each kpt in image 0, the matched kpt index in image 1 (-1 if no match)
            'gt_matches1': gt_matches1,       # [N,]: for each kpt in image 1, the matched kpt index in image 0 (-1 if no match)
            'valid0': valid0,     # For loss: boolean mask for keypoints0 that have a GT match

            'scene': self.conf.scene_name,
            'name': f"{self.conf.scene_name}/{base_name0}_{base_name1}", # Unique identifier for the pair
            # 'idx': torch.tensor(idx), # Optional, if needed for debugging or specific logic
        }

    def __len__(self):
        return len(self.items) 