import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class CurriculumLearningManager:
    """Manages curriculum learning phases and updates datasets accordingly."""

    def __init__(self, config: Dict, dataset_wrapper):
        """
        Args:
            config: The curriculum_learning section from config
            dataset_wrapper: SphereCraftDataset instance
        """

        self.enabled = config.get('enabled', False)
        self.phases = config.get('phases', [])
        self.dataset_wrapper = dataset_wrapper
        self.current_phase_idx = -1
        self.current_phase = None

        if self.enabled:
            self._validate_phases()
            logger.info(f"Curriculum Learning enabled with {len(self.phases)} phases.")
            for phase in self.phases:
                logger.info(f"  - {phase['name']}: epochs {phase['start_epoch']}-{phase['end_epoch']}, bins={phase['bins']}")
        
    def _validate_phases(self):
        """Validate that phases are properly configured."""
        if not self.phases:
            raise ValueError("Curriculum learning enabled but no phases defined")
        
        # Check phases are sorted and non-overlapping
        for i, phase in enumerate(self.phases):
            if i > 0:
                prev_phase = self.phases[i-1]
                if phase['start_epoch'] != prev_phase['end_epoch']:
                    logger.warning(
                        f"Phase gap detected: {prev_phase['name']} ends at epoch {prev_phase['end_epoch']}, "
                        f"but {phase['name']} starts at epoch {phase['start_epoch']}"
                    )
                   
    def update_for_epoch(self, epoch: int, train_loader=None, val_loader=None) -> bool:
        """
        Update datasets if entering a new phase.
        
        Args:
            epoch: Current epoch number
            train_loader: Training DataLoader (optional, will recreate if needed)
            val_loader: Validation DataLoader (optional, will recreate if needed)
        
        Returns:
            bool: True if phase changed and dataloaders need to be recreated
        """
        if not self.enabled:
            return False
        
        # Find which phase we should be in
        target_phase_idx = None
        for idx, phase in enumerate(self.phases):
            if phase['start_epoch'] <= epoch < phase['end_epoch']:
                target_phase_idx = idx
                break
        
        # Check if we're entering a new phase
        if target_phase_idx != self.current_phase_idx:
            if target_phase_idx is None:
                logger.warning(f"Epoch {epoch} is outside all defined curriculum phases")
                return False
            
            self.current_phase_idx = target_phase_idx
            self.current_phase = self.phases[target_phase_idx]
            
            logger.info(f"=" * 80)
            logger.info(f"CURRICULUM LEARNING: Entering {self.current_phase['name']}")
            logger.info(f"Epoch: {epoch}, Bins: {self.current_phase['bins']}")
            logger.info(f"Description: {self.current_phase.get('description', 'N/A')}")
            logger.info(f"=" * 80)
            
            return True
        
        return False
    
    def get_current_bins(self) -> Optional[List[str]]:
        """Get the bins that should be used for the current phase."""
        if not self.enabled or self.current_phase is None:
            return None
        return self.current_phase['bins']
    
    def get_dataloader(self, split: str, **dataloader_kwargs):
        """
        Get a dataloader with the current curriculum bins applied.
        
        Args:
            split: 'train' or 'val'
            **dataloader_kwargs: Additional arguments to pass to get_data_loader
        
        Returns:
            DataLoader configured for the current curriculum phase
        """
        current_bins = self.get_current_bins()
        
        # Get the dataset with current bins
        dataset = self.dataset_wrapper.get_dataset(split, current_bins=current_bins)
        
        # Update the dataset wrapper's internal dataset reference if needed
        # This depends on your BaseDataset implementation
        
        return self.dataset_wrapper.get_data_loader(split, **dataloader_kwargs)
