# python 3.10.8


import argparse

import numpy as np
import pandas as pd
import torch
from util.ssim import *
from util.utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--quick",
    action="store_true",
    help="Analyze 5 images spanning the sparsity range and 3 levels per distortion.",
)
args = parser.parse_args()
quick = args.quick

drop_fractions = np.linspace(0, 1.0, 20)
variance_levels = np.logspace(np.log10(0.0001), np.log10(0.1), num=20)
blur_levels = np.logspace(np.log10(0.5), np.log10(6), num=20)
pa_levels = [1, 7, 6, 5, 4, 3, 2]

PA_from_base, df_images, greyscaleImages = load_images()
base_sparsities = np.asarray(
    [calculate_sparsity(image) for image in greyscaleImages],
    dtype=float,
)

if quick:
    sparsity_order = np.argsort(base_sparsities, kind="stable")
    sample_positions = np.linspace(
        0,
        len(sparsity_order) - 1,
        num=min(5, len(sparsity_order)),
    ).round().astype(int)
    selected_indices = sparsity_order[sample_positions]
    df_images = df_images.iloc[selected_indices].reset_index(drop=True)
    greyscaleImages = [greyscaleImages[i] for i in selected_indices]
    base_sparsities = base_sparsities[selected_indices]
    selected_keys = df_images["base_key"].tolist()
    PA_from_base = {
        key: PA_from_base[key]
        for key in selected_keys
    }

    def representative_levels(levels):
        positions = np.linspace(
            0,
            len(levels) - 1,
            num=min(3, len(levels)),
        ).round().astype(int)
        return np.asarray(levels)[positions]

    drop_fractions = representative_levels(drop_fractions)
    variance_levels = representative_levels(variance_levels)
    blur_levels = representative_levels(blur_levels)
    pa_levels = representative_levels(pa_levels).tolist()

reference_image_keys = df_images["base_key"].tolist()

# 1. Pixel Dropout
avg_scores_drop, mean_taus_drop, mean_ps_drop, scores_drop, sensitivities_drop, df_drop = evaluate_metrics_on_distortions(
    imgs_ref=greyscaleImages,
    distort_fn="drop",
    levels=drop_fractions
)
print("Pixel dropout")
for variant, score in mean_taus_drop.items():
    print(f"variant: {variant}, tau score: {score}")


# 2. Noise
avg_scores_noise, mean_taus_noise, mean_ps_noise, scores_noise, sensitivities_noise, df_noise = evaluate_metrics_on_distortions(
    imgs_ref=greyscaleImages,
    distort_fn="noise",
    levels=variance_levels
)
print("\nNoise:")
for variant, score in mean_taus_noise.items():
    print(f"variant: {variant}, tau score: {score}")


# 3. Blur
avg_scores_blur, mean_taus_blur, mean_ps_blur, scores_blur, sensitivities_blur, df_blur = evaluate_metrics_on_distortions(
    imgs_ref=greyscaleImages,
    distort_fn="blur",
    levels=blur_levels
)
print("\nBlur:")
for variant, score in mean_taus_blur.items():
    print(f"variant: {variant}, tau score: {score}")


# 4. Frame Averaging (PA)
(
    avg_scores_pa,
    mean_taus_pa,
    mean_ps_pa,
    scores_pa,
    sparsities_pa,
    df_pa
) = evaluate_metrics_on_pa_distortions(
    pa_from_base=PA_from_base,
    levels=pa_levels
)
print("Frame Averaging (PA):")
for variant, score in mean_taus_pa.items():
    print(f"variant: {variant}, tau score: {score:.3f}")

dfs = []
distortions = [
    (
        "drop",
        drop_fractions,
        avg_scores_drop,
        mean_taus_drop,
        mean_ps_drop,
        scores_drop,
        sensitivities_drop,
    ),
    (
        "noise",
        variance_levels,
        avg_scores_noise,
        mean_taus_noise,
        mean_ps_noise,
        scores_noise,
        sensitivities_noise,
    ),
    (
        "blur",
        blur_levels,
        avg_scores_blur,
        mean_taus_blur,
        mean_ps_blur,
        scores_blur,
        sensitivities_blur,
    ),
    (
        "pa",
        pa_levels,
        avg_scores_pa,
        mean_taus_pa,
        mean_ps_pa,
        scores_pa,
        sparsities_pa,
    ),
]


for name, levels, avg_s, taus, ps, scores_dict, sens_dict in distortions:
    for metric_name, avg_list in avg_s.items():
        df_part = pd.DataFrame(
            {
                "distortion_type": name,
                "metric_variant": metric_name,
                "distortion_level": list(levels),
                "avg_score": avg_list,
                "mean_tau": taus.get(metric_name, None),
                "mean_p": ps.get(metric_name, None),
                "sparsity": [
                    sens_dict.get(metric_name, {}).get(i, None)
                    for i in range(len(levels))
                ],
                "per_image_scores": [
                    scores_dict.get(metric_name, {}).get(i, None)
                    for i in range(len(levels))
                ],
                "reference_image_keys": [
                    reference_image_keys
                    for _ in range(len(levels))
                ],
                "base_sparsities": [
                    base_sparsities
                    for _ in range(len(levels))
                ],
            }
        )
        dfs.append(df_part)

df_tidy = pd.concat(dfs, ignore_index=True)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def clean_val(x):
    """Recursively converts PyTorch Tensors & NumPy types into standard Python floats/lists."""
    if HAS_TORCH and isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        return x.item() if x.numel() == 1 else x.tolist()
    elif isinstance(x, (np.ndarray, np.generic)):
        return x.item() if x.size == 1 else x.tolist()
    elif isinstance(x, list):
        return [clean_val(v) for v in x]
    elif isinstance(x, dict):
        return {k: clean_val(v) for k, v in x.items()}
    return x

# Apply recursively across all object/nested columns in df_tidy
for col in df_tidy.columns:
    if df_tidy[col].dtype == "object":
        df_tidy[col] = df_tidy[col].apply(clean_val)

# Save the per-image lists with PyArrow's native nested-data support.
output_path = (
    "image_quality_metrics_analysis_quick.parquet"
    if quick
    else "image_quality_metrics_analysis.parquet"
)
df_tidy.to_parquet(output_path, engine="pyarrow")

df_tidy