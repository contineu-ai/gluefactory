import logging
import numpy as np
import torch
import os
from torch.utils.data import Dataset # Direct import is fine
from ..settings import DATA_PATH # Assuming this is where datasets are stored
from .base_dataset import BaseDataset # Crucial import

logger = logging.getLogger(__name__)

class SphereCraftDataset(BaseDataset):
    default_conf = {
        "batch_size": 1,
        "num_workers": 1,
        "prefetch_factor": 2,

        # --- Paths ---
        "data_dir": "finetuning", # Root for all SphereCraft scenes, relative to DATA_PATH
        "pair_subdir": "pairs2/easy",
        "keypoints_detector": "xfeat", # e.g., 'superpoint', 'sift'
    }

    def _init(self, conf):
        """Initialization method, called by BaseDataset."""
        self.conf = conf # self.conf is already set by BaseDataset __init__

        self.scene_root = DATA_PATH / self.conf.data_dir
        self.kpt_detector_name = self.conf.keypoints_detector
        self.pair_dir = self.scene_root / self.conf.pair_subdir
      
        if not self.pair_dir.exists():
            raise FileNotFoundError(f"Pair directory not found: {self.pair_dir}") 

        # The actual torch.utils.data.Dataset instances will be created in get_dataset
        logger.info(f"Initialized SphereCraftDataset for finetuning with detector: {self.kpt_detector_name}")


    def get_dataset(self, split):
        """Returns an instance of torch.utils.data.Dataset for the
            requested split ('train', 'val', or 'test')."""
        assert split in ["train", "val"], f"Unknown split: {split}, only train and val are accepted."
        return _PairDatasetSphereCraft(self.conf, split, self.scene_root, self.pair_dir)


class _PairDatasetSphereCraft(Dataset): # Standard PyTorch Dataset
    def __init__(self, conf, split, scene_root, pair_dir):
        self.conf = conf
        self.split = split
        self.scene_root = scene_root
        self.pair_dir = pair_dir

        self.items = []

        if split=="train":
            for pair in os.listdir(self.pair_dir):
                self.items.append(pair)
        else:
            all_pairs = sorted(os.listdir(self.pair_dir))
            num_pairs = max(1, len(all_pairs) // 8)
            self.items = list(np.random.choice(all_pairs, num_pairs, replace=False))

        if not self.items:
            logger.warning(f"No items loaded for split '{self.split}' from {self.pair_dir}.")

        logger.info(f"Loaded {len(self.items)} pairs for split '{self.split}' from {self.pair_dir}.")

    
    def __getitem__(self, idx):
        pair_name = self.items[idx]

        # The worker_init_fn in BaseDataset handles seeding if self.conf.reseed is true
        # So, no need for explicit fork_rng here unless more complex per-item seeding is needed.

        data = np.load(os.path.join(self.pair_dir, pair_name))

        # # Image data (usually needs to be float32 and channel-first)
        # image0 = torch.from_numpy(data['image0']).float()
        # image1 = torch.from_numpy(data['image1']).float()

        # Debugging for angle
        # yaw_pitch_roll_0 = data['yaw_pitch_roll_0']
        # yaw_pitch_roll_1 = data['yaw_pitch_roll_1']

        # Keypoint-related data (float32 for model input)
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

        # Image size and name (handle potential nested arrays from np.save)
        # image_size0 = torch.from_numpy(data['image_size0'])
        # image_size1 = torch.from_numpy(data['image_size1'])
        # name = str(data['name']) # Ensure name is a plain string
        name = pair_name.split('.')[0]

        # image0 = load_image(f"/mnt/d/code/glue-factory/data/finetuning/688cabe1568a518c577a0f7c/688cba040abb8edc0d488cd0/images/{name.split('_')[0]}")
        # image1 = load_image(f"/mnt/d/code/glue-factory/data/finetuning/688cabe1568a518c577a0f7c/688cba040abb8edc0d488cd0/images/{name.split('_')[1]}")
        return {
            # 'image0': image0,
            'keypoints0': keypoints0,
            'descriptors0':descriptors0,
            'scores0': scores0,
            # 'image_size0': image_size0, # not needed for normalization since using spherical coordinates

            # 'image1': image1,
            'keypoints1': keypoints1,
            'descriptors1': descriptors1,
            'scores1': scores1,
            # 'image_size1': image_size1,

            'matches': matches,           # Original [N, 2] format, if needed elsewhere
            'gt_matches0': gt_matches0,
            'gt_matches1': gt_matches1,

            'name': name # Base image used for alterations
        }

    def __len__(self):
        return len(self.items)