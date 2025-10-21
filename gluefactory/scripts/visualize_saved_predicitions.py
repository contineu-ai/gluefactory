import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from gluefactory.visualization.viz2d import plot_images, plot_keypoints, plot_matches

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
    print(f"Number of output matches: {len(gt_matches)}")

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
        plt.gcf().suptitle(f"Output Matches", fontsize=16)
        plt.savefig(f"{output_path}_matches.png")
        # plt.show() # Display the second plot


image0_path = "/data/code/glue-factory/datasets/spherecraft_data/berlin/images/00000205.jpg"
image1_path = "/data/code/glue-factory/datasets/spherecraft_data/berlin/images/00000211.jpg"

#Load predictions
pred_data = torch.load("/data/code/glue-factory/debug_predictions/iter_0.pt")

# Extract data
kpts0 = pred_data['keypoints0'][0].numpy()  # Remove batch dimension
kpts1 = pred_data['keypoints1'][0].numpy()
matches0 = pred_data['matches0'][0].numpy()
scores0 = pred_data['matching_scores0'][0].detach().numpy()

# Filter valid matches
confidence_threshold = 0.1  # Set to 0 to see all matches
valid = (matches0 > -1) & (scores0 > confidence_threshold)

mkpts0 = kpts0[valid]
mkpts1 = kpts1[matches0[valid]]
matches_indices = np.column_stack([np.where(valid)[0], matches0[valid]])

# Load images
image0_bgr = cv2.resize(cv2.imread(str(image0_path)), (1920, 960))
image1_bgr = cv2.resize(cv2.imread(str(image1_path)), (1920, 960))

output_path = "/data/code/glue-factory/debug_predictions/00000205_00000211"

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