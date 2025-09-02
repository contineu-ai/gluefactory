import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path
from omegaconf import OmegaConf

# Make sure gluefactory is in your python path
from gluefactory.models import get_model
from gluefactory.utils.xfeat_utils import generate_keypoints
from gluefactory.utils.equirectangular_utils import equirectangular_to_dicemap
from gluefactory.visualization.viz2d import plot_images, plot_keypoints, plot_matches
from torch.nn.utils.rnn import pad_sequence

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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

# --- Helper function for visualization ---
def visualize_batch_item(image0, image1, kpts0, kpts1, matches, output_path, color=None,
                       text=None, path=None, show_keypoints=False,
                       fast_viz=False, opencv_display=False,
                       opencv_title='matches'):
    """Visualizes a single item from a collated batch."""
    print(f"Scene: data_item['scene'][item_index_in_batch]")
    print(f"Pair Name: {opencv_title}")

    # --- Prepare Data ---
    img0 = image0
    img1 = image1

    # Get the keypoints (which are lists of tensors) and convert to numpy
    # Note: data_item['keypoints0'] is a LIST of tensors. We need the specific one.
    kpts0 = kpts0
    kpts1 = kpts1

    gt_matches = matches

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
    fig.suptitle(f"Keypoints for Pair", fontsize=16)
    plt.savefig(f"{output_path}_keypoints.png")
    # plt.show() # Display the first plot

    # 6. Plot the matches in a separate figure.
    # plot_matches creates its own figure with the two images and the match lines.
    if kpts0_matched.shape[0] > 0:
        # We need to create a new figure for the matches plot.
        # A simple way is to just call plot_images again, then plot_matches on top.
        plot_images([img0, img1], titles=["Image 0", "Image 1"])
        plot_matches(kpts0_matched, kpts1_matched, ps=6, lw=0.7)
        plt.gcf().suptitle(f"GT Matches", fontsize=16)
        plt.savefig(f"{output_path}_matches.png")
        # plt.show() # Display the second plot

def main(checkpoint, image0, image1, output):
    # --- Step 2: Load the trained LightGlue model ---
    print("Loading trained LightGlue model...")
    # Load the checkpoint. We load to CPU to avoid potential GPU memory issues.
    checkpoint = torch.load(str(checkpoint), map_location='cpu')

    # The checkpoint is a dictionary containing 'conf' and 'model' state_dict
    conf = OmegaConf.create(checkpoint['conf'])
    
    # Instantiate the model using the saved configuration
    # get_model is a helper from gluefactory that finds the correct model class
    model = get_model(conf.model.name)(conf.model).to(DEVICE)
    
    # Load the weights. The weights are saved under the key 'model'.
    model_weights = checkpoint['model']

    # IMPORTANT: If you trained with DDP (distributed), the keys will have a 'module.' prefix.
    # We need to remove this prefix before loading the state_dict.
    model_weights = {k.replace('module.', ''): v for k, v in model_weights.items()}
    
    model.load_state_dict(model_weights, strict=False)
    model.eval() # Set to evaluation mode
    print("Model loaded successfully.")

    # --- Step 4: Load images and extract features ---
    print("Extracting features from images...")
    # Load images as grayscale numpy arrays
    image0_bgr = cv2.imread(str(image0))
    image1_bgr = cv2.imread(str(image1))

    name = f"{Path(image0).stem}_{Path(image1).stem}"
    output_dir = Path(output)
    output_path = output_dir / f"{name}_matches.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to tensors for the feature extractor
    image0_dicemap = equirectangular_to_dicemap(image0_bgr)
    image1_dicemap = equirectangular_to_dicemap(image1_bgr)
    
    pred0 = generate_keypoints(image0_dicemap, num_keypoints=2048)
    pred1 = generate_keypoints(image1_dicemap, num_keypoints=2048)
    
    kpts0, desc0, scores0 = [torch.from_numpy(arr) for arr in pred0]
    kpts1, desc1, scores1 = [torch.from_numpy(arr) for arr in pred1]

    kpts0, kpts1 = pad_sequence([kpts0, kpts1], batch_first=True, padding_value=0, padding_side="right")
    desc0, desc1 = pad_sequence([desc0, desc1], batch_first=True, padding_value=0, padding_side="right")
    scores0, scores1 = pad_sequence([scores0, scores1], batch_first=True, padding_value=0, padding_side="right")

    print(f"Found {len(kpts0)} keypoints in image 0 and {len(kpts1)} in image 1.")

    # --- Step 5: Prepare data for LightGlue ---
    # LightGlue expects a dictionary with batched tensors.
    # We add a batch dimension of 1.
    data = {
        'keypoints0': kpts0[None].to(DEVICE), # Add batch dim and move to device
        'descriptors0': desc0[None].to(DEVICE),
        'scores0': scores0[None].to(DEVICE),
        'keypoints1': kpts1[None].to(DEVICE),
        'descriptors1': desc1[None].to(DEVICE),
        'scores1': scores1[None].to(DEVICE),
    }

    # --- Step 6: Run LightGlue inference ---
    print("Running matching with your trained LightGlue model...")
    with torch.no_grad():
        pred = model(data)

    # The prediction is a dictionary containing the matches
    # 'matches0' contains the index of the matched keypoint in image1 for each keypoint in image0
    # A value of -1 means no match was found.
    # Filter for valid matches
    matches0 = pred['matches0'].cpu().numpy()[0]  # Shape: (N,)
    scores0 = pred['matching_scores0'].cpu().numpy()[0]  # Shape: (N,)

    confidence_threshold = 0.2

    valid = (matches0 > -1) & (scores0 > confidence_threshold)
    mkpts0 = kpts0[valid] # kpts0 is already a numpy array
    mkpts1 = kpts1[matches0[valid]] # kpts1 is already a numpy array
    matches_indices = np.column_stack([np.where(valid)[0], matches0[valid]])
    
    print(f"Found {len(mkpts0)} matches.")
    
    # --- Step 7: Visualize the matches ---
    visualize_batch_item(
        cv2.cvtColor(image0_bgr, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2RGB),
        kpts0,
        kpts1,
        matches_indices,
        output_path,
        show_keypoints=False,
        opencv_title=f"Matches ({len(matches_indices)})"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test My LightGlue Model")
    parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--image0', type=str, help='Path to the first image')
    parser.add_argument('--image1', type=str, help='Path to the second image')
    parser.add_argument('--output', type=str, help='Output directory for results')

    args = parser.parse_args()
    main(
        checkpoint=args.checkpoint,
        image0=args.image0,
        image1=args.image1,
        output=args.output
    )