# test_spherecraft_loader.py
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np

# --- Adjust this path to your Glue Factory root directory ---
GLUE_FACTORY_ROOT = Path(__file__).resolve().parent.parent # Example if script is in glue_factory_root/scripts/
sys.path.insert(0, str(GLUE_FACTORY_ROOT))
# --- ---
try:
    from gluefactory.settings import DATA_PATH # To confirm where data is expected
    from gluefactory.datasets import get_dataset # The function from datasets/__init__.py
    from gluefactory.visualization.viz2d import plot_images, plot_keypoints, plot_matches # For visualization
except ImportError:
    print("Error: Could not import 'gluefactory' utilities.")
    print("Please ensure the GlueFactory library is installed and in your Python path.")
    sys.exit(1)

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

def visualize_batch_item(data_item, item_index_in_batch=0):
    """Visualizes a single item from a collated batch."""
    print(f"\n--- Visualizing item from batch (original index in batch: {item_index_in_batch}) ---")
    print(f"Scene: data_item['scene'][item_index_in_batch]")
    print(f"Pair Name: {data_item['name'][item_index_in_batch]}")

    # --- Prepare Data ---
    # Convert image tensor from CHW to HWC for plotting
    img0 = data_item['image0'][item_index_in_batch].permute(1, 2, 0).cpu().numpy()
    img1 = data_item['image1'][item_index_in_batch].permute(1, 2, 0).cpu().numpy()

    # Get the keypoints (which are lists of tensors) and convert to numpy
    # Note: data_item['keypoints0'] is a LIST of tensors. We need the specific one.
    kpts0 = data_item['keypoints0'][item_index_in_batch].cpu().numpy()
    kpts1 = data_item['keypoints1'][item_index_in_batch].cpu().numpy()

    gt_matches = data_item['matches'][item_index_in_batch].cpu().numpy()

    print(f"Image 0 shape: {img0.shape}, Keypoints 0 shape: {kpts0.shape}")
    print(f"Image 1 shape: {img1.shape}, Keypoints 1 shape: {kpts1.shape}")
    print(f"Number of valid GT matches: {len(gt_matches)}")

    # --- Get Matched Keypoints for Visualization ---
    kpts0_matched = kpts0[gt_matches[:, 0]]
    kpts1_matched = kpts1[gt_matches[:, 1]]

    kpts0_matched = standard_spherical_to_pixel(kpts0_matched, img0.shape[1], img0.shape[0])
    kpts1_matched = standard_spherical_to_pixel(kpts1_matched, img1.shape[1], img1.shape[0])

    # --- Visualization Workflow ---

    # 1. Plot the two images side-by-side. This creates a new figure and axes.
    # The plot_images function will handle creating a 1x2 grid.
    plot_images([img0, img1], titles=["Image 0 (processed)", "Image 1 (processed)"])

    # 2. Get the axes that plot_images just created.
    # plt.gcf() gets the "current figure", and .axes gets its axes.
    fig = plt.gcf()
    axes = fig.axes

    # 3. Plot all keypoints on these axes.
    # plot_keypoints expects a list of keypoint arrays and a list of axes.
    plot_keypoints([kpts0, kpts1], axes=axes, colors='lime', ps=6)

    # 4. Highlight the matched keypoints in a different color.
    if kpts0_matched.shape[0] > 0:
        plot_keypoints([kpts0_matched, kpts1_matched], axes=axes, colors='red', ps=8)

    # 5. Add a title to the whole figure.
    fig.suptitle(f"Keypoints for Pair: {data_item['name'][item_index_in_batch]}", fontsize=16)
    plt.show() # Display the first plot

    # 6. Plot the matches in a separate figure.
    # plot_matches creates its own figure with the two images and the match lines.
    if kpts0_matched.shape[0] > 0:
        # We need to create a new figure for the matches plot.
        # A simple way is to just call plot_images again, then plot_matches on top.
        plot_images([img0, img1], titles=["Image 0", "Image 1"])
        plot_matches(kpts0_matched, kpts1_matched, ps=6, lw=0.5)
        plt.gcf().suptitle(f"GT Matches for {data_item['name'][item_index_in_batch]}", fontsize=16)
        plt.show() # Display the second plot


if __name__ == "__main__":
    print(f"Expecting SphereCraft data within: {DATA_PATH}")

    # 2. Instantiate the top-level dataset class
    # The `get_dataset` function from datasets/__init__.py should find 'spherecraft'
    # and return the SphereCraftDataset class.
    SphereCraftDatasetClass = get_dataset("spherecraft")
    spherecraft_dataset_manager = SphereCraftDatasetClass({})


    # 3. Get a specific split (e.g., 'train' or 'val')
    # This calls `spherecraft_dataset_manager.get_dataset('train')` which returns
    # an instance of `_PairDatasetSphereCraft`.
    print("\n--- Getting 'train' dataset ---")
    try:
        train_torch_dataset = spherecraft_dataset_manager.get_dataset("train")
        print(f"Successfully got 'train' dataset. Number of items: {len(train_torch_dataset)}")
    except Exception as e:
        print(f"ERROR getting 'train' dataset: {e}")
        sys.exit(1)

    # 4. (Optional but Recommended) Test with DataLoader for batching and workers
    print("\n--- Testing DataLoader ---")
    # Use the get_data_loader method from the dataset_manager
    # We are not using distributed training here, so distributed=False
    try:
        train_loader = spherecraft_dataset_manager.get_data_loader("train", shuffle=True, distributed=False)
        print(f"Successfully created DataLoader for 'train' split.")
    except Exception as e:
        print(f"ERROR creating DataLoader: {e}")

    # 5. Iterate and inspect a few batches/items
    num_batches_to_check = 1
    num_items_to_visualize_per_batch = 1

    print(f"Number of batches in train_loader: {len(train_loader)}")

    for i, batch_data in enumerate(train_loader):
        if i >= num_batches_to_check:
            break
        print(f"\n--- Batch {i+1} ---")
        print("Batch Keys:", batch_data.keys())
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: type={type(value)}, shape={value.shape}, dtype={value.dtype}, device={value.device}")
            elif isinstance(value, list) and value and isinstance(value[0], str): # List of strings (names, scenes)
                 print(f"  {key}: type={type(value)}, len={len(value)}, example='{value[0]}'")
            else:
                print(f"  {key}: type={type(value)}")

        # Visualize the first item of this batch
        if batch_data['image0'].shape[0] > 0: # If batch is not empty
             visualize_batch_item(batch_data, item_index_in_batch=0)
        else:
            print("Batch is empty, skipping visualization.")

    # # 6. (More targeted) Get a single item directly for detailed debugging
    # print("\n--- Testing single item direct access (item 0) ---")
    # if len(train_torch_dataset) > 0:
    #     try:
    #         single_item_data = train_torch_dataset[0] # Calls _PairDatasetSphereCraft.__getitem__(0)
    #         print("Single Item Keys:", single_item_data.keys())
    #         for key, value in single_item_data.items():
    #              if isinstance(value, torch.Tensor):
    #                 print(f"  {key}: type={type(value)}, shape={value.shape}, dtype={value.dtype}")
    #              else:
    #                 print(f"  {key}: type={type(value)}, value='{value}'")
            
    #         # To visualize a single item NOT from a batch, we need to unsqueeze the tensors
    #         # to simulate a batch of size 1 for the visualize_batch_item function
    #         single_item_batch_sim = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v]) for k,v in single_item_data.items()}
    #         visualize_batch_item(single_item_batch_sim, item_index_in_batch=0)

    #     except Exception as e:
    #         print(f"ERROR getting/processing single item 0: {e}")
    #         import traceback
    #         traceback.print_exc()
    # else:
    #     print("Train dataset is empty, cannot test single item access.")

    print("\n--- Test script finished ---")
