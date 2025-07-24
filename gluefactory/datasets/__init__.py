import importlib.util
import logging

from ..utils.tools import get_class
from .base_dataset import BaseDataset

from .spherecraft import SphereCraftDataset
from .spherecraft_pretraining import SphereCraftPretrainingDataset

logger = logging.getLogger(__name__) # Add logger

# Pre-register known datasets
AVAILABLE_DATASETS = {
    'spherecraft': SphereCraftDataset,
    'spherecraft_pretraining': SphereCraftPretrainingDataset,
    # Add other known datasets here, e.g.
    # 'megadepth': MegaDepth, (if you import it)
    # 'hpatches': HPatches,
}

def get_dataset(name):
    if name in AVAILABLE_DATASETS:
        return AVAILABLE_DATASETS[name]

    # Fallback to dynamic import for less common datasets or custom paths
    # (Original logic from your __init__.py)
    import_paths = [name, f"gluefactory.datasets.{name}", f"{__name__}.{name}"]
    for path in import_paths:
        try:
            spec = importlib.util.find_spec(path)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            try:
                # Try to get the class directly
                cls = get_class(path, BaseDataset)
                logger.info(f"Dynamically loaded dataset '{name}' from path '{path}' as class '{cls.__name__}'.")
                return cls
            except AssertionError: # Not a class or not subclass of BaseDataset
                try:
                    # Try to import as a module and look for __main_dataset__
                    mod = __import__(path, fromlist=[""]) # fromlist=[""] is key for relative imports if path is like ".submodule"
                    main_dataset_attr = getattr(mod, '__main_dataset__', None)
                    if main_dataset_attr and issubclass(main_dataset_attr, BaseDataset):
                        logger.info(f"Dynamically loaded dataset '{name}' from path '{path}' via __main_dataset__ attribute.")
                        return main_dataset_attr
                    else:
                        logger.warning(f"Module '{path}' found for dataset '{name}', but no suitable class or __main_dataset__ attribute found.")
                except Exception as exc: # Catch broader import/attribute errors
                    logger.warning(f"Error trying to load dataset '{name}' from module '{path}': {exc}")
                    continue
        else:
            logger.debug(f"No module found for dataset '{name}' at import path '{path}'.")

    raise RuntimeError(f'Dataset {name} not found in known datasets or any of [{" ".join(import_paths)}]')