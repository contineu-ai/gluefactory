import logging
import numpy as np
import torch
import json
import os
from torch.utils.data import Dataset # Direct import is fine
from ..settings import DATA_PATH # Assuming this is where datasets are stored
from .base_dataset import BaseDataset # Crucial import
from ..utils.image import load_image

logger = logging.getLogger(__name__)

class SphereCraftDataset(BaseDataset):
    default_conf = {
        "batch_size": 1,
        "num_workers": 1,
        "prefetch_factor": 2,

        # --- Paths ---
        "data_dir": "/data/code/glue-factory/data/finetuning", # Root for all SphereCraft scenes, relative to DATA_PATH
        "pair_subdir": "finetuning_pairs_spherecraft",
        "keypoints_detector": "xfeat", # e.g., 'superpoint', 'sift'

        # --- Curriculum Learning ---
        "bin_files": {
            "easy": "bin_easy_by_matches.json",
            "medium": "bin_medium_by_matches.json",
            "hard": "bin_hard_by_matches.json"
        }
    }

    def _init(self, conf):
        """Initialization method, called by BaseDataset."""
        self.conf = conf # self.conf is already set by BaseDataset __init__

        self.scene_root = DATA_PATH / self.conf.data_dir
        self.kpt_detector_name = self.conf.keypoints_detector
        self.pair_dir = self.scene_root / self.conf.pair_subdir
      
        if not self.pair_dir.exists():
            raise FileNotFoundError(f"Pair directory not found: {self.pair_dir}") 

        # Load bin files for curriculum learning
        self.bins = {}
        if hasattr(self.conf, 'bin_files'):
            for bin_name, bin_path in self.conf.bin_files.items():
                full_path = self.scene_root / bin_path
                if full_path.exists():
                    with open(full_path, 'r') as f:
                        self.bins[bin_name] = set(json.load(f))
                    logger.info(f"Loaded {len(self.bins[bin_name])} pairs for bin '{bin_name}'")
                else:
                    logger.warning(f"Bin file not found: {full_path}")
                    self.bins[bin_name] = set()

        # The actual torch.utils.data.Dataset instances will be created in get_dataset
        logger.info(f"Initialized SphereCraftDataset for finetuning with detector: {self.kpt_detector_name}")


    def get_dataset(self, split, current_bins=None):
        """Returns an instance of torch.utils.data.Dataset for the
            requested split ('train', 'val', or 'test')."""
        assert split in ["train", "val"], f"Unknown split: {split}, only train and val are accepted."
        return _PairDatasetSphereCraft(
            self.conf, 
            split, 
            self.scene_root, 
            self.pair_dir,
            self.bins,
            current_bins
            )


class _PairDatasetSphereCraft(Dataset): # Standard PyTorch Dataset
    def __init__(self, conf, split, scene_root, pair_dir, bins, current_bins=None):
        self.conf = conf
        self.split = split
        self.scene_root = scene_root
        self.pair_dir = pair_dir
        self.bins = bins
        self.current_bins = current_bins

        self._load_items()

    def _load_items(self):
        """Load items based on split and current curriculum bins."""

        all_files = sorted(os.listdir(self.pair_dir))
        all_pairs = sorted([f for f in all_files if f.endswith('.npz')])
        logger.info(f"Found {len(all_pairs)} .npz files in {self.pair_dir}")

        # Filter by curriculum bins if specified
        if self.current_bins is not None and len(self.bins) > 0:
            allowed_pairs = set()
            
            # Collect all allowed pairs from current bins
            for bin_name in self.current_bins:
                if bin_name in self.bins:
                    allowed_pairs.update(self.bins[bin_name])

            # DEBUG: Print filtering info
            # logger.info(f"DEBUG: Current bins: {self.current_bins}")
            # logger.info(f"DEBUG: Allowed pairs from bins: {allowed_pairs}")
            # logger.info(f"DEBUG: Checking overlap...")

            if allowed_pairs:
                # Filter to only pairs that are in allowed_pairs
                filtered_pairs = [p for p in all_pairs if p in allowed_pairs]
                logger.info(f"DEBUG: After filtering: {len(filtered_pairs)} pairs match bins")
                logger.info(f"Filtered to {len(filtered_pairs)} pairs using bins: {self.current_bins}")
                all_pairs = filtered_pairs
            else:
                logger.warning(f"No allowed pairs found for bins {self.current_bins}")

        # Split into train/val
        if self.split=="train":
            self.items = all_pairs
        else:  # val
            num_pairs = max(1, len(all_pairs) // 4)
            if len(all_pairs) > 0:
                self.items = list(np.random.choice(all_pairs, min(num_pairs, len(all_pairs)), replace=False))
            else:
                self.items = []

        if not self.items:
            logger.warning(f"No items loaded for split '{self.split}' from {self.pair_dir}.")

        logger.info(f"Loaded {len(self.items)} pairs for split '{self.split}' from {self.pair_dir}.")

    def update_bins(self, new_bins):
        """Update te curriculum bins and reload items"""
        self.current_bins = new_bins
        self._load_items()
        logger.info(f"Updated dataset to use bins: {new_bins}, now has {len(self.items)} items.")

    def __getitem__(self, idx):
        pair_name = self.items[idx]
        data = np.load(os.path.join(self.pair_dir, pair_name))

        keypoints0 = torch.from_numpy(data['keypoints0']).float()
        descriptors0 = torch.from_numpy(data['descriptors0']).float()
        scores0 = torch.from_numpy(data['scores0']).float()

        keypoints1 = torch.from_numpy(data['keypoints1']).float()
        descriptors1 = torch.from_numpy(data['descriptors1']).float()
        scores1 = torch.from_numpy(data['scores1']).float()

        # Ground truth match data (int64 / long for indexing)
        matches = torch.from_numpy(data['matches']).long()
        gt_matches0 = torch.from_numpy(data['gt_matches0']).long()
        gt_matches1 = torch.from_numpy(data['gt_matches1']).long()

     
        return {
            'keypoints0': keypoints0,
            'descriptors0':descriptors0,
            'scores0': scores0,

            'keypoints1': keypoints1,
            'descriptors1': descriptors1,
            'scores1': scores1,

            'matches': matches,           # Original [N, 2] format, if needed elsewhere
            'gt_matches0': gt_matches0,
            'gt_matches1': gt_matches1,
        }

    def __len__(self):
        return len(self.items)