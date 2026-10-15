# python 3.10.8
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mahotas.thresholding import bernsen
from PIL import Image as PIL_Image
from skimage import img_as_float
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import threshold_local, threshold_otsu
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util import img_as_float

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      


def load_images():
    # 1. Crop and preprocess images into .npy
    output_folder = "./cropped_grayscale"
    root_dir = "./OneDrive_1_5-19-2025"


    os.makedirs(output_folder, exist_ok=True)

    uncropped_images = []
    root_dir = "./OneDrive_1_5-19-2025"

    for image_path in sorted(
        glob.glob(os.path.join(root_dir, "**/*.png"), recursive=True)
    ):
        uncropped_images.append(imread(image_path))

        with PIL_Image.open(image_path) as img:
            width, height = img.size
            crop_box = (0, int(height * 0.1), width, height)
            cropped_img = img.crop(crop_box)

            # Build a filename preserving full folder hierarchy
            rel_path = os.path.relpath(image_path, start=root_dir)
            rel_parts = rel_path.split(os.sep)
            folder_part = "__".join(rel_parts[:-1])
            orig_filename = rel_parts[-1]

            if folder_part:
                new_filename = f"{folder_part}__{orig_filename}"
            else:
                new_filename = orig_filename

            grayscale_img = img_as_float(rgb2gray(rgba2rgb(cropped_img)))
            np.save(
                os.path.join(
                    output_folder, new_filename.replace(".png", ".npy")
                ),
                grayscale_img,
            )

    # 2. Identify bases that have complete sequence up to PA7
    all_files = sorted(
        f
        for f in os.listdir(output_folder)
        if os.path.isfile(os.path.join(output_folder, f))
    )

    pa7_base_list = set()
    for fname in all_files:
        if "PA7" in fname:
            pa7_base_list.add(fname.split("__PA7")[0])

    # 3. Build dictionary mapping and construct structured Metadata DataFrame
    records = []
    PA_from_base = dict()

    for base_key in sorted(pa7_base_list):
        PA_from_base[base_key] = dict()

        # Load PA sequence
        for i in range(1, 8):
            PA_from_base[base_key][f"PA{i}"] = np.load(
                os.path.join(output_folder, f"{base_key}__PA{i}.npy")
            )

        # Reconstruct original source metadata
        rel_path = base_key.replace("__", os.sep)
        original_filename = f"{rel_path.split(os.sep)[-1]}__PA1.png"
        subfolder_path = os.path.dirname(rel_path)

        records.append(
            {
                "base_key": base_key,
                "relative_subfolder": subfolder_path,
                "reference_filename": original_filename,
                "full_source_path": os.path.join(
                    root_dir, rel_path + "__PA1.png"
                ),
                "greyscale_image": PA_from_base[base_key]["PA1"],
            }
        )

    # 4. Create final metadata DataFrame
    df_images = pd.DataFrame(records)

    # Extract reference images list directly from DataFrame
    greyscaleImages = df_images["greyscale_image"].tolist()
    greyscaleImages

    return PA_from_base, df_images, greyscaleImages


def mask_square_over_roi(image: np.ndarray, roi_mask: np.ndarray, distortion_level: float = 0.25):
    """
    Applies a zero-mask square block centered around a position overlapping the ROI.
    If no overlap is found, places the block centered at the bottom-right.

    Args:
        image (np.ndarray): Float image (H, W) or (H, W, C), values in [0, 1].
        roi_mask (np.ndarray): Binary mask (H, W), where 1 indicates ROI.
        distortion_level (float): Fraction of image area to mask.

    Returns:
        np.ndarray: Image with a masked block.
        (int, int, int): (x, y, square_size) — top-left coordinates and size.
    """
    assert image.shape[:2] == roi_mask.shape, "ROI mask must match image shape"
    assert 0 <= distortion_level <= 1.0, "Distortion level must be between 0 and 1"
    assert np.any(roi_mask), "ROI mask is empty (all zeros)."

    h, w = image.shape[:2]
    block_area = distortion_level * h * w
    square_size = int(np.sqrt(block_area))
    square_size = min(square_size, h, w)  # Ensure block fits in image

    result = image.copy()

    y_start = int(0.1 * h)
    x_start = int(0.1 * w)

    # Try to find first position with any overlap with ROI
    for y in range(y_start, h - square_size):
        for x in range(x_start, w - square_size):
            if roi_mask[y,x]:  # Partial overlap allowed
                # Center the square around (x, y)
                cx = x + square_size // 2
                cy = y + square_size // 2
                top_left_x = max(0, min(w - square_size, cx - square_size // 2))
                top_left_y = max(0, min(h - square_size, cy - square_size // 2))

                if image.ndim == 2:
                    result[top_left_y:top_left_y+square_size, top_left_x:top_left_x+square_size] = 0.0
                else:
                    result[top_left_y:top_left_y+square_size, top_left_x:top_left_x+square_size, :] = 0.0

                return result, (top_left_x, top_left_y, square_size)

    # Fallback: center at bottom-right
    cx = w - 1
    cy = h - 1
    top_left_x = max(0, min(w - square_size, cx - square_size // 2))
    top_left_y = max(0, min(h - square_size, cy - square_size // 2))

    if image.ndim == 2:
        result[top_left_y:top_left_y+square_size, top_left_x:top_left_x+square_size] = 0.0
    else:
        result[top_left_y:top_left_y+square_size, top_left_x:top_left_x+square_size, :] = 0.0

    return result, (top_left_x, top_left_y, square_size)



def calculate_combined_mask(img1, img2, sensitivity=0.48, and_or="or"):
    patch_size = 2 * (np.array(img1.shape) // 16) + 1

    # Adaptive thresholding (Gaussian mean)
    block_size = int(np.mean(patch_size))
    if block_size % 2 == 0:
        block_size += 1

    # Compute adaptive thresholds
    mean_thresh1 = threshold_local(img1, block_size, method='gaussian', offset=-(0.5 - sensitivity))
    mean_thresh2 = threshold_local(img2, block_size, method='gaussian', offset=-(0.5 - sensitivity))

    mask1 = img1 > mean_thresh1
    mask2 = img2 > mean_thresh2
    Mask = np.logical_or(mask1, mask2)

    return Mask

def calculate_single_mask(img1, sensitivity=0.5, and_or="or"):
    patch_size = 2 * (np.array(img1.shape) // 16) + 1

    # Adaptive thresholding (Gaussian mean)
    block_size = int(np.mean(patch_size))
    if block_size % 2 == 0:
        block_size += 1

    # Compute adaptive thresholds
    mean_thresh1 = threshold_local(img1, block_size, method='gaussian', offset=-(0.5 - sensitivity))

    mask1 = img1 > mean_thresh1

    return mask1


def calculate_sparsity(image):
    """Return the fraction of pixels whose value is exactly zero."""
    return 1.0 - np.count_nonzero(image) / image.size


def drop_pixels(img, drop_fraction=0.5, seed=42):
    """
    Randomly zeroes out a fraction of pixels from the image.
    
    Parameters:
        img: 2D numpy array (grayscale)
        drop_fraction: fraction of pixels to drop (e.g., 0.3 = 30%)
        seed: optional random seed for reproducibility
        
    Returns:
        distorted_img: image with pixels dropped (set to 0)
        mask: binary mask indicating kept pixels
    """
    np.random.seed(seed)
    mask = np.random.rand(*img.shape) > drop_fraction
    distorted = img * mask
    return distorted

def snr_reference(ref_img, test_img): # signal is determined by ground truth noise determined by distorted img
    signal_power = np.var(ref_img)
    noise_power = np.var(ref_img - test_img)
    return 10 * np.log10(signal_power / (noise_power + 1e-12))



def get_distortion_vars(df, distortion_type):
    d = df[df["distortion_type"] == distortion_type].copy()

    # Preserve the order of distortion levels
    d = d.sort_values("distortion_level")

    avg_scores = {}
    mean_taus = {}
    mean_ps = {}
    scores = {}
    sensitivities = {}

    for metric_name, g in d.groupby("metric_variant", sort=False):
        avg_scores[metric_name] = g["avg_score"].tolist()

        # mean_tau / mean_p are constant across distortion levels
        # in the tidy representation, so take the first non-null value.
        tau_values = g["mean_tau"].dropna()
        p_values = g["mean_p"].dropna()

        mean_taus[metric_name] = tau_values.iloc[0] if len(tau_values) else None
        mean_ps[metric_name] = p_values.iloc[0] if len(p_values) else None

        sensitivities[metric_name] = {
            i: value
            for i, value in enumerate(g["sparsity"].tolist())
        }

        scores[metric_name] = {
            i: value
            for i, value in enumerate(g["per_image_scores"].tolist())
        }

    levels = d["distortion_level"].drop_duplicates().tolist()

    return (
        levels,
        avg_scores,
        mean_taus,
        mean_ps,
        scores,
        sensitivities,
    )


import numpy as np
from matplotlib.gridspec import GridSpec


def compute_metrics(original, distorted, data_range=1.0):
    return {
        'SSIM': structural_similarity(original, distorted, data_range=data_range),
        'PSNR': peak_signal_noise_ratio(original, distorted, data_range=data_range),
        'RMSE': rmse(original, distorted),
        "S3IM": s3im(original, distorted)[0],
        "MS-SSIM": msssim(original, distorted),
        "MS-S3IM": mss3im(original, distorted)[0],
        "CW-SSIM": np.average(cw_ssim(original, distorted)),
        "CW-S3IM": s3im(original, distorted, SSIM_variant=cw_ssim)[0],
        "SNR": snr_reference(original, distorted)
    }

# --- small helper: format a 1x2 image panel with titles containing scores ---
def _image_panel(fig, gs_cell, original, distorted, title_left="Original", title_right="Distorted", compute_metrics_fn=None):
    sub = gs_cell.subgridspec(1, 2, wspace=0.05)
    axL = fig.add_subplot(sub[0, 0])
    axR = fig.add_subplot(sub[0, 1])

    axL.imshow(original, cmap='gray', vmin=0, vmax=1)
    axL.axis('off')
    axL.set_title(title_left, fontsize=11)

    axR.imshow(distorted, cmap='gray', vmin=0, vmax=1)
    axR.axis('off')
    if compute_metrics_fn is not None:
        mets = compute_metrics_fn(original, distorted)
        axR.set_title(
            (f"{title_right}\n"
             f"PSNR {mets['PSNR']:.2f} dB  |  RMSE {mets['RMSE']:.4f}\n"
             f"SSIM {mets['SSIM']:.4f}  |  S3IM {mets['S3IM']:.4f}\n"
             f"MS-SSIM {mets['MS-SSIM']:.4f}  |  MS-S3IM {mets['MS-S3IM']:.4f}\n"
             f"CW-SSIM {mets['CW-SSIM']:.4f}  |  CW-S3IM {mets['CW-S3IM']:.4f}"),
            fontsize=10
        )
    else:
        axR.set_title(title_right, fontsize=11)

# --- small helper: draw the multi-axis curves on the right side ---
def _curves_panel(fig, gs_cell, x, avg_scores, xlabel, title, invert_x=False):
    ax1 = fig.add_subplot(gs_cell)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('IQA Scores', color='blue')

    # IQA families on left y-axis
    l1, = ax1.plot(x, avg_scores['ssim'],  'o-', label='SSIM',   color='tab:blue')
    l2, = ax1.plot(x, avg_scores['wssim'], 'o-', label='CW-SSIM',color='gold')
    l3, = ax1.plot(x, avg_scores['msssim'],'o-', label='MS-SSIM',color='black')
    l4, = ax1.plot(x, avg_scores['s3im'],  's-', label='S3IM',   color='tab:green')
    l5, = ax1.plot(x, avg_scores['ws3im'], 's-', label='CW-S3IM',color='tab:orange')
    l6, = ax1.plot(x, avg_scores['mss3im'],'s-', label='MS-S3IM',color='saddlebrown')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 1)

    # PSNR on first right axis
    ax2 = ax1.twinx()
    l7, = ax2.plot(x, avg_scores['psnr'], 'x-', label='PSNR', color='tab:red')
    ax2.set_ylabel('PSNR (dB)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 80)

    # (1 - RMSE) on second right axis (offset outward)
    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("outward", 60))
    inv_rmse = [1.0 - (v**0.5) for v in avg_scores['mse']]
    l8, = ax3.plot(x, inv_rmse, '^-', label='1 - RMSE', color='tab:purple')
    ax3.set_ylabel('1 - RMSE', color='purple')
    ax3.tick_params(axis='y', labelcolor='purple')
    ax3.set_ylim(0, 1)

    if invert_x:
        ax1.invert_xaxis()
        ax2.invert_xaxis()
        ax3.invert_xaxis()

    ax1.set_title(title)

    # combine legends (put under the panel)
    lines = [l1,l2,l3,l4,l5,l6,l7,l8]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=4, fontsize=9)

# --- MASTER: build one figure with rows = {Dropout, Blur, Noise}, cols = {examples, curves} ---
def plot_examples_and_curves_side_by_side(
    # example images
    original_img,
    dropped_img,
    blur_img,
    noisy_img,
    # level arrays and average-score dicts
    drop_x, avg_scores_drop,
    blur_x, avg_scores_blur,
    snr_x,  avg_scores_noise,
    compute_metrics_fn
):
    fig = plt.figure(figsize=(18, 16), constrained_layout=True)
    gs = GridSpec(nrows=3, ncols=2, width_ratios=[1, 1.6], height_ratios=[1, 1, 1], figure=fig)

    # Row 1: Dropout
    _image_panel(fig, gs[0, 0], original_img, dropped_img,
                 title_left="Original", title_right="Pixel Dropout (example)",
                 compute_metrics_fn=compute_metrics_fn)
    _curves_panel(fig, gs[0, 1], drop_x, avg_scores_drop,
                  xlabel="Drop Fraction", title="Pixel Dropout — Metrics vs Drop Fraction")

    # Row 2: Blur
    _image_panel(fig, gs[1, 0], original_img, blur_img,
                 title_left="Original", title_right="Gaussian Blur (example)",
                 compute_metrics_fn=compute_metrics_fn)
    _curves_panel(fig, gs[1, 1], blur_x, avg_scores_blur,
                  xlabel="Sigma", title="Gaussian Blur — Metrics vs Sigma")

    # Row 3: Noise (SNR-based, invert x so worse noise on right)
    _image_panel(fig, gs[2, 0], original_img, noisy_img,
                 title_left="Original", title_right="Additive Gaussian Noise (example)",
                 compute_metrics_fn=compute_metrics_fn)
    _curves_panel(fig, gs[2, 1], snr_x, avg_scores_noise,
                  xlabel="Signal-to-Noise Ratio (linear)", title="Noise — Metrics vs SNR (higher is cleaner)",
                  invert_x=True)

    # overall title
    fig.suptitle("Examples and Metric Trends per Distortion Type", fontsize=18, y=0.995)
    plt.show()

import numpy as np


def compute_differences(scores_dict, sparsity_dict, relative=True):
    """
    Compute differences between baseline SSIM-type metrics and their ROI-aware counterparts.
    
    relative=True -> use relative difference ( (roi - base) / base )
    relative=False -> use absolute difference (roi - base)
    """
    diffs = {}

    for (m1, m2, label) in [
        ("ssim", "s3im", "SSIM vs S3IM"),
        ("msssim", "mss3im", "MS-SSIM vs MS-S3IM"),
        ("wssim", "ws3im", "CW-SSIM vs CW-S3IM"),
    ]:
        diffs[label] = {"sparsity": [], "diff": [], "level": []}

        for level in scores_dict[m1].keys():
            base_vals = np.array(scores_dict[m1][level])
            roi_vals = np.array(scores_dict[m2][level])
            sparsity_vals = np.array(sparsity_dict[m1][level])

            if relative:
                diff_vals = np.abs(roi_vals - base_vals) / np.clip(base_vals, 1e-6, None)
            else:
                diff_vals = np.abs(roi_vals - base_vals)

            diffs[label]["sparsity"].extend(sparsity_vals)
            diffs[label]["diff"].extend(diff_vals)
            diffs[label]["level"].extend([level] * len(diff_vals))

    return diffs


def plot_differences(diffs, title_suffix="(noise)"):
    for label, data in diffs.items():
        plt.figure(figsize=(8,6))

        x = np.array(data["sparsity"])
        y = np.array(data["diff"])
        levels = np.array(data["level"])

        # Plot each distortion level separately
        for lvl in sorted(set(levels)):
            mask = levels == lvl
            x_lvl, y_lvl = x[mask], y[mask]

            # Scatter for this distortion level
            plt.scatter(x_lvl, y_lvl, alpha=0.6, s=22, label=f"Level {lvl}")

            # Regression line if there is variability in sparsity
            if np.std(x_lvl) > 0:
                a, b = np.polyfit(x_lvl, y_lvl, 1)
                xs = np.linspace(x_lvl.min(), x_lvl.max(), 200)
                plt.plot(xs, a*xs+b, linestyle="--")

        plt.title(f"Effect of Sparsity on {label} {title_suffix}")
        plt.xlabel("Image sparsity (zero-pixel fraction)")
        plt.ylabel("Relative difference in score")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
                   ncol=min(5, len(set(levels))))
        plt.tight_layout()
        plt.show()



def visualize_maps(maps, titles=None, scores=None, cmap='gray', figsize=(14, 4), ncols=5, scene_title=""):
    import string

    import matplotlib.pyplot as plt
    import numpy as np

    n = len(maps)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    if titles is None:
        titles = [f"Downscaling level {i}" for i in range(n)]
    if scores is None:
        scores = [None] * n

    # Hide any extra axes
    for ax in axes[n:]:
        ax.set_visible(False)

    for idx, (img, title, score) in enumerate(zip(maps, titles, scores)):
        ax = axes[idx]

        # ✅ Fix: squeeze singleton dimensions like (1,1,H,W) → (H,W)
        img = np.squeeze(img)

        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.axis('off')

        # Title + optional score
        if score is not None:
            ax.set_title(f"{title}\nScore: {score:.4f}", fontsize=13)
        else:
            ax.set_title(title, fontsize=13)

        # Add panel label (A, B, C, …)
        label = string.ascii_uppercase[idx]
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
                ha='left', va='top', fontsize=13, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='square,pad=0.15'))
    
        # print(img.mean())

    if scene_title:
        fig.text(0.5, 0.95, scene_title, ha='center', va='bottom',
                 fontsize=15, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec


# ---------- helpers for panel labels ----------
def _labels_from(start='A', count=3):
    base = ord(start.upper())
    labels = [chr(base + i) for i in range(count)]
    next_label = chr(base + count)
    return labels, next_label

def _panel_tag(ax, label, x=0.015, y=0.985):
    ax.text(x, y, label, transform=ax.transAxes,
            ha='left', va='top', fontsize=14, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2))

# --- helper: 1×2 image panel with metric titles on the right image ---
def _image_panel(fig, gs_cell, original, distorted,
                 title_left="Original", title_right="Distorted",
                 compute_metrics_fn=None):
    sub = gs_cell.subgridspec(1, 2, wspace=0.05)
    axL = fig.add_subplot(sub[0, 0])
    axR = fig.add_subplot(sub[0, 1])

    axL.imshow(original, cmap='gray', vmin=0, vmax=1)
    axL.axis('off')
    axL.set_title(title_left, fontsize=10)

    axR.imshow(distorted, cmap='gray', vmin=0, vmax=1)
    axR.axis('off')
    if compute_metrics_fn is not None:
        mets = compute_metrics_fn(original, distorted)
        axR.set_title(
            (f"{title_right}\n"
             f"PSNR {mets['PSNR']:.2f} dB  |  RMSE {mets['RMSE']:.4f}\n"
             f"SSIM {mets['SSIM']:.4f}  |  S3IM {mets['S3IM']:.4f}\n"
             f"MS-SSIM {mets['MS-SSIM']:.4f}  |  MS-S3IM {mets['MS-S3IM']:.4f}\n"
             f"CW-SSIM {mets['CW-SSIM']:.4f}  |  CW-S3IM {mets['CW-S3IM']:.4f}"),
            fontsize=10
        )
    else:
        axR.set_title(title_right, fontsize=10)

    return axL, axR  # <- return axes so we can tag them

# --- helper: multi-axis curves panel (left y: IQA, right y: PSNR, offset y: 1-RMSE) ---
def _curves_panel(fig, gs_cell, x, avg_scores, xlabel, title, invert_x=False):
    ax1 = fig.add_subplot(gs_cell)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('IQA Scores & RMSE - 1', color='blue')

    l1, = ax1.plot(x, avg_scores['ssim'],   'o-', label='SSIM',    color='tab:blue')
    l2, = ax1.plot(x, avg_scores['wssim'],  'o-', label='CW-SSIM', color='gold')
    l3, = ax1.plot(x, avg_scores['msssim'], 'o-', label='MS-SSIM', color='black')
    l4, = ax1.plot(x, avg_scores['s3im'],   's-', label='S3IM',    color='tab:green')
    l5, = ax1.plot(x, avg_scores['ws3im'],  's-', label='CW-S3IM', color='tab:orange')
    l6, = ax1.plot(x, avg_scores['mss3im'], 's-', label='MS-S3IM', color='saddlebrown')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    l7, = ax2.plot(x, avg_scores['psnr'], 'x-', label='PSNR', color='tab:red')
    ax2.set_ylabel('PSNR (dB)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 80)

    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("outward", 60))
    inv_rmse = [1.0 - (v**0.5) for v in avg_scores['mse']]
    l8, = ax3.plot(x, inv_rmse, '^-', label='1 - RMSE', color='tab:purple')
    ax3.set_ylabel('1 - RMSE', color='purple')
    ax3.tick_params(axis='y', labelcolor='purple')
    ax3.set_ylim(0, 1)

    if invert_x:
        ax1.invert_xaxis(); ax2.invert_xaxis(); ax3.invert_xaxis()

    ax1.set_title(title)
    lines = [l1,l2,l3,l4,l5,l6,l7,l8]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.17),
               ncol=4, fontsize=9)

    return ax1  # <- return the primary plot axes to tag

# --- single-row figure: images (left) + curves (right) ---
def plot_single_row_match_image_height(original_img, distorted_img,
                                       x_levels, avg_scores, xlabel, title_curves,
                                       compute_metrics_fn=None,
                                       dpi=100, width_ratio=(2.2, 2.0),
                                       show_image_text=True,
                                       invert_x=False,
                                       start_label='A'):
    """
    Makes curve panel same height as images; adds panel labels A/B/C.
    Returns the next label after those used (so you can keep counting).
    """
    # Figure size driven by image height
    H, W = original_img.shape[:2]
    width_in  = ( (width_ratio[0] + width_ratio[1]) * W / 2 ) / dpi * 1.1
    height_in = H / dpi / 1.2

    fig = plt.figure(figsize=(width_in, height_in), constrained_layout=True)
    gs = GridSpec(nrows=1, ncols=2, width_ratios=width_ratio, figure=fig)

    # left: images
    sub = gs[0, 0].subgridspec(1, 2, wspace=0.03)
    axL = fig.add_subplot(sub[0, 0])
    axL.imshow(original_img, cmap='gray', vmin=0, vmax=1, aspect='equal')
    axL.axis('off')
    if show_image_text:
        axL.set_title("Original", fontsize=9)

    axR = fig.add_subplot(sub[0, 1])
    axR.imshow(distorted_img, cmap='gray', vmin=0, vmax=1, aspect='equal')
    axR.axis('off')
    if show_image_text and compute_metrics_fn is not None:
        mets = compute_metrics_fn(original_img, distorted_img)
        axR.set_title(
            (f"{xlabel}\n"
             f"PSNR {mets['PSNR']:.2f} dB  |  RMSE {mets['RMSE']:.4f}\n"
             f"SSIM {mets['SSIM']:.4f}  |  S3IM {mets['S3IM']:.4f}\n"
             f"MS-SSIM {mets['MS-SSIM']:.4f}  |  MS-S3IM {mets['MS-S3IM']:.4f}\n"
             f"CW-SSIM {mets['CW-SSIM']:.4f}  |  CW-S3IM {mets['CW-S3IM']:.4f}"),
            fontsize=11
        )

    # right: curves
    axPlot = _curves_panel(fig, gs[0, 1], x_levels, avg_scores,
                           xlabel=xlabel, title=title_curves, invert_x=invert_x)

    # ---- add panel labels A/B/C ----
    labels, next_label = _labels_from(start=start_label, count=3)
    _panel_tag(axL,   labels[0])   # A
    _panel_tag(axR,   labels[1])   # B
    _panel_tag(axPlot, labels[2])  # C

    # neat margins
    plt.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.22)
    plt.show()
    return next_label

def plot_row(ax_row, titles, imgs, scores, row_labels):
    for idx, (ax, title, img, score, lab) in enumerate(zip(ax_row, titles, imgs, scores, row_labels)):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, aspect='auto')
        ax.axis("off")
        ax.set_box_aspect(1)  # square subplot cell

        # Per-image label at top-left
        ax.text(
            0.02, 0.98, lab,
            transform=ax.transAxes,
            color='white', fontsize=16, fontweight='bold', va='top', ha='left',
            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2')
        )

        # Titles (include metric when available)
        if score is not None:
            ax.set_title(f"{title}\n{title.split()[0]}: {score:.4f}", fontsize=16)
        else:
            ax.set_title(title, fontsize=16)

# ============================================================
# Absolute difference utilities
# ============================================================

def absolute_diff(base_vals, roi_vals, mode="raw", eps=1e-6):
    """
    Compute different types of absolute differences between two
    sets of scores.

    Parameters
    ----------
    base_vals : array-like
        Baseline scores.
    roi_vals : array-like
        Comparison scores.
    mode : str
        One of:
            "raw"
            "headroom_abs"
            "spd_abs"
            "logit_abs"
            "asin_abs"
    eps : float
        Numerical stability constant.
    """
    base_vals = np.asarray(base_vals, dtype=float)
    roi_vals = np.asarray(roi_vals, dtype=float)

    raw = np.abs(roi_vals - base_vals)

    if mode == "raw":
        return raw

    if mode == "headroom_abs":
        up = roi_vals >= base_vals

        denom = np.empty_like(base_vals)

        denom[up] = np.maximum(
            1.0 - base_vals[up],
            eps
        )

        denom[~up] = np.maximum(
            base_vals[~up],
            eps
        )

        return raw / denom

    if mode == "spd_abs":
        return np.abs(
            2 * (roi_vals - base_vals)
            / np.clip(roi_vals + base_vals, eps, None)
        )

    if mode == "logit_abs":
        p_b = np.clip(base_vals, eps, 1 - eps)
        p_r = np.clip(roi_vals, eps, 1 - eps)

        def logit(p):
            return np.log(p) - np.log(1 - p)

        return np.abs(
            logit(p_r) - logit(p_b)
        )

    if mode == "asin_abs":
        def tr(p):
            return np.arcsin(
                np.sqrt(np.clip(p, eps, 1 - eps))
            )

        return np.abs(
            tr(roi_vals) - tr(base_vals)
        )

    raise ValueError(f"Unknown mode: {mode}")


# ============================================================
# Plot ordering
# ============================================================

PAIR_ORDER = [
    "SSIM vs S3IM",
    "MS-SSIM vs MS-S3IM",
]


# ============================================================
# Compute differences
# ============================================================

def compute_differences(
    scores_dict,
    sparsity_vec,
    exclude_indices=(),
    level_step=7,
    level_start=3,
):
    """
    Compute absolute differences between:

        SSIM   vs S3IM
        MS-SSIM vs MS-S3IM

    for selected distortion levels.

    Parameters
    ----------
    scores_dict : dict
        Dictionary structured approximately as:

            {
                "ssim": {
                    level_index: [score_per_image, ...]
                },
                "s3im": {
                    level_index: [score_per_image, ...]
                },
                "msssim": {
                    level_index: [score_per_image, ...]
                },
                "mss3im": {
                    level_index: [score_per_image, ...]
                }
            }

    sparsity_vec : array-like
        One sparsity value per image.

    exclude_indices : iterable
        Image indices to exclude.

    level_step : int
        Take every `level_step` distortion level.

    level_start : int
        Starting index into the available distortion levels.
        `3` reproduces your original `[3::7]` behavior.
    """

    diffs = {}

    pairs = [
        ("ssim", "s3im", "SSIM vs S3IM"),
        ("msssim", "mss3im", "MS-SSIM vs MS-S3IM"),
    ]

    sparsity_vec = np.asarray(
        sparsity_vec,
        dtype=float
    )

    exclude_indices = np.asarray(
        list(exclude_indices),
        dtype=int
    )

    for m1, m2, label in pairs:

        diffs[label] = {
            "sparsity": [],
            "diff": [],
            "level": [],
        }

        # ----------------------------------------------------
        # Check that both metrics exist
        # ----------------------------------------------------

        if m1 not in scores_dict:
            raise KeyError(
                f"'{m1}' not found in scores_dict. "
                f"Available keys: {list(scores_dict.keys())}"
            )

        if m2 not in scores_dict:
            raise KeyError(
                f"'{m2}' not found in scores_dict. "
                f"Available keys: {list(scores_dict.keys())}"
            )

        # ----------------------------------------------------
        # Select distortion levels
        # ----------------------------------------------------

        available_levels = list(
            scores_dict[m1].keys()
        )

        selected_levels = available_levels[
            level_start::level_step
        ]

        # ----------------------------------------------------
        # Process each distortion level
        # ----------------------------------------------------

        for level in selected_levels:

            if level not in scores_dict[m2]:
                continue

            base_vals = np.asarray(
                scores_dict[m1][level],
                dtype=float
            )

            roi_vals = np.asarray(
                scores_dict[m2][level],
                dtype=float
            )

            s_vals = np.asarray(
                sparsity_vec,
                dtype=float
            )

            # ------------------------------------------------
            # Align lengths
            # ------------------------------------------------

            n0 = min(
                len(base_vals),
                len(roi_vals),
                len(s_vals),
            )

            if n0 == 0:
                continue

            base_vals = base_vals[:n0]
            roi_vals = roi_vals[:n0]
            s_vals = s_vals[:n0]

            # ------------------------------------------------
            # Remove specified image indices
            # ------------------------------------------------

            valid_exclusions = exclude_indices[
                (exclude_indices >= 0)
                & (exclude_indices < n0)
            ]

            keep = ~np.isin(
                np.arange(n0),
                valid_exclusions
            )

            base_vals = base_vals[keep]
            roi_vals = roi_vals[keep]
            s_vals = s_vals[keep]

            # ------------------------------------------------
            # Final alignment
            # ------------------------------------------------

            n = min(
                len(base_vals),
                len(roi_vals),
                len(s_vals),
            )

            if n == 0:
                continue

            base_vals = base_vals[:n]
            roi_vals = roi_vals[:n]
            s_vals = s_vals[:n]

            # ------------------------------------------------
            # Absolute difference
            # ------------------------------------------------

            diff_vals = np.abs(
                base_vals - roi_vals
            )

            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            diffs[label]["sparsity"].extend(
                s_vals.tolist()
            )

            diffs[label]["diff"].extend(
                diff_vals.tolist()
            )

            diffs[label]["level"].extend(
                [level] * n
            )

    return diffs


# ============================================================
# Color utilities
# ============================================================

def _level_color_map(levels):
    """
    Assign a stable color to each distortion level.
    """

    levels = sorted(levels)

    cmap = plt.get_cmap("tab20")

    return {
        level: cmap(i % 20)
        for i, level in enumerate(levels)
    }


# ============================================================
# Panel labels
# ============================================================

def _panel_tag(
    ax,
    label,
    x=0.015,
    y=0.985,
):
    """
    Add A/B/C/... panel label.
    """

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
            pad=2,
        ),
    )


# ============================================================
# Six-panel plot
# ============================================================

def plot_all_six(
    diffs_drop,
    diffs_noise,
    diffs_blur,
    poly_deg=2,
    outlier_threshold=3.5,
):
    """
    Create a 3 x 2 figure.

    Rows:
        Dropout
        Noise
        Blur

    Columns:
        SSIM vs S3IM
        MS-SSIM vs MS-S3IM

    Each distortion level gets:
        - scatter points
        - polynomial fit

    Severe y-outliers are excluded per distortion level using the
    modified z-score (median absolute deviation).

    A shared legend is placed underneath.
    """

    row_titles = [
        "(drop)",
        "(noise)",
        "(blur)",
    ]

    dist_diffs = [
        diffs_drop,
        diffs_noise,
        diffs_blur,
    ]

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 12),
        sharey=True,
    )

    panel_labels = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    ]

    panel_idx = 0

    legend_handles = {}
    total_outliers_removed = 0

    # --------------------------------------------------------
    # Gather all distortion levels
    # --------------------------------------------------------

    all_levels_union = set()

    for diffs in dist_diffs:

        for d in diffs.values():

            all_levels_union.update(
                np.unique(d["level"])
            )

    all_levels_union = sorted(
        all_levels_union
    )

    color_map = _level_color_map(
        all_levels_union
    )

    # --------------------------------------------------------
    # Plot each panel
    # --------------------------------------------------------

    for r, diffs in enumerate(dist_diffs):

        for c, label in enumerate(PAIR_ORDER):

            ax = axes[r, c]

            data = diffs[label]

            x = np.asarray(
                data["sparsity"],
                dtype=float
            )

            y = np.asarray(
                data["diff"],
                dtype=float
            )

            levels = np.asarray(
                data["level"]
            )

            # ------------------------------------------------
            # Plot each distortion level
            # ------------------------------------------------

            for lvl in all_levels_union:

                mask = (
                    levels == lvl
                )

                if not np.any(mask):
                    continue

                x_lvl = x[mask]
                y_lvl = y[mask]

                finite = np.isfinite(x_lvl) & np.isfinite(y_lvl)
                x_lvl = x_lvl[finite]
                y_lvl = y_lvl[finite]

                if not len(y_lvl):
                    continue

                median = np.median(y_lvl)
                mad = np.median(np.abs(y_lvl - median))
                if mad > 0:
                    modified_z = (
                        0.6744897501960817
                        * np.abs(y_lvl - median)
                        / mad
                    )
                    keep = modified_z <= outlier_threshold
                    total_outliers_removed += int(np.count_nonzero(~keep))
                    x_lvl = x_lvl[keep]
                    y_lvl = y_lvl[keep]

                if not len(y_lvl):
                    continue

                # --------------------------------------------
                # Scatter
                # --------------------------------------------

                sc = ax.scatter(
                    x_lvl,
                    y_lvl,
                    s=16,
                    alpha=0.7,
                    color=color_map[lvl],
                    label=f"Level {lvl}",
                )

                if lvl not in legend_handles:
                    legend_handles[lvl] = sc

                # --------------------------------------------
                # Polynomial fit
                # --------------------------------------------

                if (
                    len(x_lvl) >= poly_deg + 1
                    and np.unique(x_lvl).size
                    >= poly_deg + 1
                ):

                    try:

                        coeffs = np.polyfit(
                            x_lvl,
                            y_lvl,
                            deg=poly_deg,
                        )

                        xs = np.linspace(
                            x_lvl.min(),
                            x_lvl.max(),
                            300,
                        )

                        ys = np.polyval(
                            coeffs,
                            xs,
                        )

                        ax.plot(
                            xs,
                            ys,
                            linewidth=1.5,
                            color=color_map[lvl],
                        )

                    except np.linalg.LinAlgError:
                        pass

            # ------------------------------------------------
            # Titles
            # ------------------------------------------------

            ax.set_title(
                f"{label} {row_titles[r]}"
            )

            # ------------------------------------------------
            # Axis labels
            # ------------------------------------------------

            if r == 2:
                ax.set_xlabel(
                    "Image sparsity (zero-pixel fraction)"
                )

            if c == 0:
                ax.set_ylabel(
                    "Absolute difference in score"
                )

            # ------------------------------------------------
            # Grid
            # ------------------------------------------------

            ax.grid(
                True,
                alpha=0.2,
            )

            # ------------------------------------------------
            # Panel label
            # ------------------------------------------------

            _panel_tag(
                ax,
                panel_labels[panel_idx],
            )

            panel_idx += 1

    print(
        f"Removed {total_outliers_removed} severe outlier(s) "
        f"(modified z-score > {outlier_threshold}) from the plot."
    )

    # --------------------------------------------------------
    # Shared legend
    # --------------------------------------------------------

    if legend_handles:

        legend_levels = [
            lvl
            for lvl in all_levels_union
            if lvl in legend_handles
        ]

        fig.legend(
            [
                legend_handles[lvl]
                for lvl in legend_levels
            ],
            [
                f"Level {lvl}"
                for lvl in legend_levels
            ],
            loc="lower center",
            ncol=min(
                8,
                len(legend_levels),
            ),
            bbox_to_anchor=(0.5, 0.015),
            frameon=False,
            fontsize=9,
        )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    plt.tight_layout(
        rect=[
            0,
            0.05,
            1,
            1,
        ]
    )

    plt.show()



def visualize_metric_maps(original_img, shifted_img,
                          SSIM_score, ssim_map,
                          cw_ssim_score, cw_ssim_map,
                          ms_ssim_score, ms_ssim_map,
                          S3IM_score, s3im_mask, s3im_map,
                          cw_s3im_score, cw_s3im_map,
                          ms_s3im_score, ms_s3im_map,
                          scene_title=""):
    """Return titles, images, and scores aligned for plotting"""
    titles = [
        "Original", 
        "Shifted", 
        "ROI Mask",
        "SSIM Map",
        "CW-SSIM Map", 
        "MS-SSIM Map",
        "S3IM Map",
        "CW-S3IM Map", 
        "MS-S3IM Map"
    ]
    scores = [
        None, None, None,
        SSIM_score,
        cw_ssim_score,
        ms_ssim_score,
        S3IM_score,
        cw_s3im_score,
        ms_s3im_score,
    ]
    images = [
        original_img,
        shifted_img,
        s3im_mask,
        ssim_map,
        cw_ssim_map,
        ms_ssim_map,
        s3im_map,
        cw_s3im_map,
        ms_s3im_map,
    ]
    return titles, images, scores, scene_title


import matplotlib


def visualize_multiple_masks(image, masks, titles, alpha=0.6):
    """
    Display the original image with multiple mask overlays side-by-side.
    
    Args:
        image: Grayscale image (2D array)
        masks: List of masks (boolean 2D arrays)
        titles: List of titles (one per mask)
        alpha: Transparency level for the mask overlay
    """
    n = len(masks)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if (type(axes) == matplotlib.axes._axes.Axes):
        axes = [axes]
    for ax, mask, title in zip(axes, masks, titles):
        rgb_image = np.stack([image, image, image], axis=-1)
        overlay = rgb_image.copy()
        overlay[mask] = [1, 0, 0]  # Red where masked
        blended = (1 - alpha) * rgb_image + alpha * overlay

        ax.imshow(blended)
        ax.set_title(title, fontsize=32)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_mask_overlay(image, mask, alpha=0.6, title="S3IM Mask Overlay"):
    # Create an RGB version of the grayscale image
    rgb_image = np.stack([image, image, image], axis=-1)

    # Create red overlay where mask == True
    overlay = rgb_image.copy()
    overlay[mask] = [1, 0, 0]  # Bright Red where mask is True

    # Blend the two images
    blended = (1 - alpha) * rgb_image + alpha * overlay

    plt.figure(figsize=(6, 6))
    plt.imshow(blended)
    plt.title(title)
    plt.axis('off')
    plt.show()


import numpy as np


def calculate_and_visualise_masks(img1, img2):
    mask_local = calculate_combined_mask(img1, img2)

    mask1 = bernsen(img1, radius=3, contrast_threshold=20)
    mask2 = bernsen(img2, radius=3, contrast_threshold=20)
    mask_bernsen = np.logical_or(mask1, mask2)

    thresh1 = threshold_otsu(img1)
    thresh2 = threshold_otsu(img2)
    mask1 = img1 > thresh1
    mask2 = img2 > thresh2
    mask_otsu = np.logical_or(mask1, mask2)

    visualize_multiple_masks(img2, [mask_local*0, mask_local, mask_bernsen, mask_otsu], alpha=0.3, titles=["Original", "AGT",  "Bernsen", "Otsu"])


import string

import numpy as np


def compute_masks_for_pair(img1, img2):
    """Compute the four masks + titles for one image pair."""
    mask_local = calculate_combined_mask(img1, img2)

    # Bernsen
    mask1_b = bernsen(img1, radius=2, contrast_threshold=5)
    mask2_b = bernsen(img2, radius=2, contrast_threshold=5)
    mask_bernsen = np.logical_or(mask1_b, mask2_b)

    # Otsu
    thresh1 = threshold_otsu(img1)
    thresh2 = threshold_otsu(img2)
    mask1_o = img1 > thresh1
    mask2_o = img2 > thresh2
    mask_otsu = np.logical_or(mask1_o, mask2_o)

    masks = [mask_local * 0, mask_local, mask_bernsen, mask_otsu]
    titles = ["Original", "AGT", "Bernsen", "Otsu"]
    return img2, masks, titles

import numpy as np


def _panel_label(idx: int) -> str:
    letters = string.ascii_uppercase
    s = ""
    n = idx
    while True:
        s = letters[n % 26] + s
        n = n // 26 - 1
        if n < 0:
            break
    return s


def visualize_all_mask_rows(dataset_rows, alpha=0.6):
    """
    dataset_rows = [
        ("Dataset Name", [
            ("HIGH", img1, img2),
            ("LOW",  img1, img2),
        ]),
        ...
    ]
    """

    total_rows = sum(len(levels) for _, levels in dataset_rows)
    n_cols = 4  # Original, AGT, Bernsen, Otsu

    fig, axes = plt.subplots(total_rows, n_cols, figsize=(4 * n_cols, 3.5 * total_rows))
    axes = np.atleast_2d(axes)

    panel_idx = 0
    row_idx = 0

    for dataset_name, levels in dataset_rows:

        for level_idx, (level_name, img1, img2) in enumerate(levels):

            base_img, masks, titles = compute_masks_for_pair(img1, img2)

            # Convert grayscale → RGB once per row  
            base_rgb = np.stack([base_img, base_img, base_img], axis=-1)

            # =====================
            # Left-side labels
            # =====================

            # dataset label (only on first row of dataset)
            if level_idx == 0:
                y = 1 - (row_idx + 0.50) / total_rows
                fig.text(
                    0.01, y,
                    dataset_name.upper(),
                    ha="left", va="center",
                    fontsize=15, fontweight="bold"
                )

            # HIGH / LOW
            # y = 1 - (row_idx + 0.50) / total_rows
            # fig.text(
            #     0.06, y,
            #     level_name.upper(),
            #     ha="left", va="center",
            #     fontsize=13, fontweight="bold"
            # )


            # =====================
            # Draw 4 columns
            # =====================
            for col in range(n_cols):
                ax = axes[row_idx, col]

                # mask for this column
                mask = masks[col]

                # Create RED overlay
                overlay = base_rgb.copy()
                overlay[mask] = [1, 0, 0]  # pure red where mask is True

                # Blend
                blended = (1 - alpha) * base_rgb + alpha * overlay

                ax.imshow(blended, vmin=0, vmax=1)
                ax.axis('off')

                # titles on top row only
                if row_idx == 0:
                    ax.set_title(titles[col], fontsize=16)

                # Panel label A, B, C, …
                label = _panel_label(panel_idx)
                ax.text(
                    0.02, 0.98, label,
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=12, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
                )

                panel_idx += 1

            row_idx += 1

    plt.tight_layout(rect=[0.17, 0, 1, 1])
    plt.subplots_adjust(wspace=0.04)
    plt.show()    




def plot_results_and_calculate_kendall(
        levels,
        avg_ssim_scores,
        avg_wssim_scores,
        avg_msssim_scores,
        avg_s3im_scores,
        avg_ws3im_scores,
        avg_mss3im_scores,
        avg_mse_scores,
        avg_psnr_scores,
        distortion_type,
        distortion_level
):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left Y-axis: SSIM and S3IM
    ax1.set_xlabel(distortion_level)
    ax1.set_ylabel('IQA Scores', color='blue')
    ax1.plot(levels, avg_ssim_scores, 'o-', label='SSIM', color='tab:blue')
    ax1.plot(levels, avg_wssim_scores, 'o-', label='CW-SSIM', color='yellow')
    ax1.plot(levels, avg_msssim_scores, 'o-', label='MS-SSIM', color='black')
    ax1.plot(levels, avg_mss3im_scores, 'o-', label='MS-S3IM', color='tab:brown')
    ax1.plot(levels, avg_s3im_scores, 's-', label='S3IM', color='tab:green')
    ax1.plot(levels, avg_ws3im_scores, 's-', label='CW-S3IM', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 1)

    # First right Y-axis: PSNR
    ax2 = ax1.twinx()
    ax2.set_ylabel('PSNR (dB)', color='red')
    ax2.plot(levels, avg_psnr_scores, 'x-', label='PSNR', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 80)

    # Second right Y-axis (MSE) — using offset
    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("outward", 60))  # Offset to create third axis
    ax3.set_ylabel('1 - RMSE', color='purple')
    ax3.plot(levels, [1-(x**0.5) for x in avg_mse_scores], '^-', label='RMSE', color='tab:purple')
    ax3.tick_params(axis='y', labelcolor='purple')
    ax3.set_ylim(0, 1)

    # Title and layout
    plt.title(f'Image Quality Metrics Under {distortion_type}')
    fig.tight_layout()

    # Combine all legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    fig.legend(lines_1 + lines_2 + lines_3, labels_1 + labels_2 + labels_3, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.08),)
    # plt.gca().invert_xaxis()
#     plt.xscale('log')
    plt.grid(True)
    plt.show()


def plot_results_snr_based(
        snr_linear_levels,
        avg_ssim_scores,
        avg_wssim_scores,
        avg_msssim_scores,
        avg_s3im_scores,
        avg_ws3im_scores,
        avg_mss3im_scores,
        avg_mse_scores,
        avg_psnr_scores,
        distortion_type,
        distortion_level
):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # X-axis is SNR (linear)
    ax1.set_xlabel(distortion_level)
    ax1.set_ylabel('Quality Metric Scores', color='blue')
    ax1.plot(snr_linear_levels, avg_ssim_scores, 'o-', label='SSIM', color='tab:blue')
    ax1.plot(snr_linear_levels, avg_wssim_scores, 'o-', label='CW-SIM', color='yellow')
    ax1.plot(snr_linear_levels, avg_msssim_scores, 'o-', label='MS-SSIM', color='black')
    ax1.plot(snr_linear_levels, avg_mss3im_scores, 'o-', label='MS-S3IM', color='tab:brown')
    ax1.plot(snr_linear_levels, avg_s3im_scores, 's-', label='S3IM', color='tab:green')
    ax1.plot(snr_linear_levels, avg_ws3im_scores, 's-', label='CW-S3IM', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 1)

    # First right Y-axis: PSNR
    ax2 = ax1.twinx()
    ax2.set_ylabel('PSNR (dB)', color='red')
    ax2.plot(snr_linear_levels, avg_psnr_scores, 'x-', label='PSNR', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 80)

    # Second right Y-axis (MSE) — using offset
    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("outward", 60))
    ax3.set_ylabel('1 - RMSE', color='purple')
    ax3.plot(snr_linear_levels, [1-(x**0.5) for x in avg_mse_scores], '^-', label='RMSE', color='tab:purple')
    ax3.tick_params(axis='y', labelcolor='purple')
    ax3.set_ylim(0, 1)

    plt.title(f'Image Quality Metrics Under {distortion_type}')
    fig.tight_layout()

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    fig.legend(lines_1 + lines_2 + lines_3, labels_1 + labels_2 + labels_3, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.08),)
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.show()