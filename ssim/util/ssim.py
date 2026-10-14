# python 3.10.8
import argparse
import glob
import sys
from math import exp

import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn.functional as F
from PIL import ImageOps
from pytorch_msssim import ms_ssim
from scipy.ndimage import gaussian_filter, shift
from scipy.stats import kendalltau
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)
from skimage.util import random_noise
from ssim.compat import Image, ImageOps
from ssim.utils import convolve_gaussian_2d, get_gaussian_kernel, to_grayscale
from util.utils import *

from ssim import compat


class SSIMImage:
    """Wraps a PIL Image object with SSIM state.

    Attributes:
      img: Original PIL Image.
      img_gray: grayscale Image.
      img_gray_squared: squared img_gray.
      img_gray_mu: img_gray convolved with gaussian kernel.
      img_gray_mu_squared: squared img_gray_mu.
      img_gray_sigma_squared: img_gray convolved with gaussian kernel -
                              img_gray_mu_squared.
    """
    def __init__(self, img, gaussian_kernel_1d=None, size=None):
        """Create an SSIMImage.

        Args:
          img (str or PIL.Image): PIL Image object or file name.
          gaussian_kernel_1d (np.ndarray, optional): Gaussian kernel
          that was generated with utils.get_gaussian_kernel is used
          to precompute common objects for SSIM computation
          size (tuple, optional): New image size to resize image to.
        """
        # Use existing or create a new PIL.Image
        self.img = img if not isinstance(img, compat.basestring) \
            else compat.Image.open(img)

        # Resize image if size is defined and different
        # from original image
        if size and size != self.img.size:
            self.img = self.img.resize(size, Image.LANCZOS)

        # Set the size of the image
        self.size = self.img.size # This will be (width, height)

        # If gaussian kernel is defined we create
        # common SSIM objects
        if gaussian_kernel_1d is not None:

            self.gaussian_kernel_1d = gaussian_kernel_1d

            # np.array of grayscale and alpha image
            self.img_gray, self.img_alpha = to_grayscale(self.img)
            if self.img_alpha is not None:
                self.img_gray[self.img_alpha == 255] = 0

            # Squared grayscale
            self.img_gray_squared = self.img_gray ** 2

            # Convolve grayscale image with gaussian
            self.img_gray_mu = convolve_gaussian_2d(
                self.img_gray, self.gaussian_kernel_1d)

            # Squared mu
            self.img_gray_mu_squared = self.img_gray_mu ** 2

            # Convolve squared grayscale with gaussian
            self.img_gray_sigma_squared = convolve_gaussian_2d(
                self.img_gray_squared, self.gaussian_kernel_1d)

            # Substract squared mu
            self.img_gray_sigma_squared -= self.img_gray_mu_squared

        # If we don't define gaussian kernel, we create
        # common CW-SSIM objects
        else:
            # Grayscale PIL.Image
            self.img_gray = ImageOps.grayscale(self.img)

class SSIM:
    """Computes SSIM between two images."""
    def __init__(self, img, gaussian_kernel_1d=None, size=None,
                 l=255, k_1=0.01, k_2=0.03, k=0.01):
        """Create an SSIM object.

        Args:
          img (str or PIL.Image): Reference image to compare other images to.
          l, k_1, k_2 (float): SSIM configuration variables.
          k (float): CW-SSIM configuration variable (default 0.01)
          gaussian_kernel_1d (np.ndarray, optional): Gaussian kernel
          that was generated with utils.get_gaussian_kernel is used
          to precompute common objects for SSIM computation
          size (tuple, optional): resize the image to the tuple size
        """
        self.k = k
        # Set k1,k2 & c1,c2 to depend on L (width of color map).
        self.c_1 = (k_1 * l) ** 2
        self.c_2 = (k_2 * l) ** 2
        self.gaussian_kernel_1d = gaussian_kernel_1d
        self.img = SSIMImage(img, gaussian_kernel_1d, size)

    def ssim_value(self, target):
        """Compute the SSIM value from the reference image to the target image.

        Args:
          target (str or PIL.Image): Input image to compare the reference image
          to. This may be a PIL Image object or, to save time, an SSIMImage
          object (e.g. the img member of another SSIM object).

        Returns:
          Computed SSIM float value.
        """
        # Performance boost if handed a compatible SSIMImage object.
        if not isinstance(target, SSIMImage) \
            or not np.array_equal(self.gaussian_kernel_1d,
                                   target.gaussian_kernel_1d):
            target = SSIMImage(target, self.gaussian_kernel_1d, self.img.size)

        img_mat_12 = self.img.img_gray * target.img_gray
        img_mat_sigma_12 = convolve_gaussian_2d(
            img_mat_12, self.gaussian_kernel_1d)
        img_mat_mu_12 = self.img.img_gray_mu * target.img_gray_mu
        img_mat_sigma_12 = img_mat_sigma_12 - img_mat_mu_12

        # Numerator of SSIM
        num_ssim = ((2 * img_mat_mu_12 + self.c_1) *
                    (2 * img_mat_sigma_12 + self.c_2))

        # Denominator of SSIM
        den_ssim = (
            (self.img.img_gray_mu_squared + target.img_gray_mu_squared +
             self.c_1) *
            (self.img.img_gray_sigma_squared +
             target.img_gray_sigma_squared + self.c_2))

        ssim_map = num_ssim / den_ssim
        index = np.average(ssim_map)
        return index

    def cw_ssim_map(self, target, width=30) -> np.ndarray: # Changed return type hint
        """Compute the complex wavelet SSIM (CW-SSIM) map from the reference
        image to the target image.

        Args:
          target (str or PIL.Image): Input image to compare the reference image
          to. This may be a PIL Image object or, to save time, an SSIMImage
          object (e.g. the img member of another SSIM object).
          width: width for the wavelet convolution (default: 30)

        Returns:
          np.ndarray: A 2D NumPy array representing the CW-SSIM map,
                      with dimensions (Height, Width) of the original image.
        """
        if not isinstance(target, SSIMImage):
            target = SSIMImage(target, size=self.img.size)

        # Define a width for the wavelet convolution
        widths = np.arange(1, width + 1)

        # Use the image data as arrays
        # .getdata() flattens the image into a 1D sequence of pixels
        sig1 = np.asarray(self.img.img_gray.getdata())
        sig2 = np.asarray(target.img_gray.getdata())

        # Convolution (Continuous Wavelet Transform)
        cwtmatr1, freqs1 = pywt.cwt(sig1, widths, 'mexh')
        cwtmatr2, freqs2 = pywt.cwt(sig2, widths, 'mexh')

        # Compute the first term
        c1c2 = np.multiply(np.abs(cwtmatr1), np.abs(cwtmatr2)) # Added np.abs
        c1_2 = np.square(np.abs(cwtmatr1)) # Added np.abs
        c2_2 = np.square(np.abs(cwtmatr2)) # Added np.abs
        num_ssim_1 = 2 * np.sum(c1c2, axis=0) + self.k
        den_ssim_1 = np.sum(c1_2, axis=0) + np.sum(c2_2, axis=0) + self.k

        # Compute the second term
        c1c2_conj = np.multiply(cwtmatr1, np.conjugate(cwtmatr2))
        num_ssim_2 = 2 * np.abs(np.sum(c1c2_conj, axis=0)) + self.k
        den_ssim_2 = 2 * np.sum(np.abs(c1c2_conj), axis=0) + self.k

        # Construct the per-pixel result (still 1D at this point)
        ssim_map_1d = (num_ssim_1 / den_ssim_1) * (num_ssim_2 / den_ssim_2)

        # Reshape the 1D ssim_map back to 2D image dimensions
        # self.img.size is (width, height), so reshape needs (height, width)
        img_width, img_height = self.img.size
        ssim_map_2d = ssim_map_1d.reshape((img_height, img_width))

        return ssim_map_2d # Return the 2D map

def main():
    """Main function for pyssim."""

    description = '\n'.join([
        'Compares an image with a list of images using the SSIM metric.',
        '  Example:',
        '    pyssim test-images/test1-1.png "test-images/*"'
    ])

    parser = argparse.ArgumentParser(
        prog='pyssim', formatter_class=argparse.RawTextHelpFormatter,
        description=description)
    parser.add_argument('--cw', help='compute the complex wavelet SSIM',
                        action='store_true')
    parser.add_argument(
        'base_image', metavar='image1.png', type=argparse.FileType('r'))
    parser.add_argument(
        'comparison_images', metavar='image path with* or image2.png')
    parser.add_argument('--width', type=int, default=None,
                        help='scales the image before computing SSIM')
    parser.add_argument('--height', type=int, default=None,
                        help='scales the image before computing SSIM')

    args = parser.parse_args()

    if args.width and args.height:
        size = (args.width, args.height)
    else:
        size = None

    if not args.cw:
        gaussian_kernel_sigma = 1.5
        gaussian_kernel_width = 11
        gaussian_kernel_1d = get_gaussian_kernel(
            gaussian_kernel_width, gaussian_kernel_sigma)

    comparison_images = glob.glob(args.comparison_images)
    is_a_single_image = len(comparison_images) == 1

    for comparison_image in comparison_images:

        if args.cw:
            ssim = SSIM(args.base_image.name, size=size)
            # This will now return the 2D map
            ssim_map = ssim.cw_ssim_value(comparison_image)
            # If you still need the average, you can compute it here:
            ssim_value = np.average(ssim_map)
        else:
            ssim = SSIM(args.base_image.name, gaussian_kernel_1d, size=size)
            ssim_value = ssim.ssim_value(comparison_image)

        if is_a_single_image:
            sys.stdout.write('%.7g' % ssim_value) # This line still prints the average
        else:
            sys.stdout.write('%s - %s: %.7g' % (
                args.base_image.name, comparison_image, ssim_value))
        sys.stdout.write('\n')

def cw_ssim(img1, img2):
    img_ref_pil = Image.fromarray((img1 * 255).astype(np.uint8)).convert('L')
    img_dist_pil = Image.fromarray((img2 * 255).astype(np.uint8)).convert('L')

    width, height = img_ref_pil.size
    # Instantiate the SSIM class with the reference image
    # For CW-SSIM, gaussian_kernel_1d is None, as per the original code
    ssim_calculator = SSIM(img_ref_pil, size=(width, height))


    # Get the CW-SSIM map
    return ssim_calculator.cw_ssim_map(img_dist_pil)


def s3im(img1, img2, sensitivity=0.5, window_size=7, SSIM_variant=structural_similarity, ms=False, increased_sensitivity=False, increase_sensitivity=True):
    """
    Sparse SSIM (S3IM) implementation in Python
    Inputs:
        img1: test image (grayscale, float)
        img2: reference image (grayscale, float)
        sensitivity: float between 0 and 1 (default 0.5)
    Outputs:
        S3IM_val: scalar SSIM over sparse regions
        S3IM_map: SSIM map (masked)
        Mask: binary mask of significant regions
    """
    Mask = calculate_combined_mask(img1, img2, sensitivity)
    if np.sum(Mask) == 0:
        if increase_sensitivity == True:
            # TODO fallback strategy
            print('inc (1): ', sensitivity)
            increased_sensitivity = True
            return s3im(img1, img2, sensitivity=sensitivity+0.1, window_size=window_size, SSIM_variant=SSIM_variant, increase_sensitivity=increase_sensitivity, increased_sensitivity=increased_sensitivity)
        else:
            return np.nan, 0, False

    if SSIM_variant == structural_similarity:
        # Use standard SSIM
        _, ssim_map = SSIM_variant(img1, img2, full=True, win_size=window_size, data_range=1.0)
        S3IM_map = ssim_map * Mask

    elif SSIM_variant == cw_ssim:
        # Use wavelet-based SSIM (W-SSIM)
        cw_ssim_map = SSIM_variant(img1, img2)


        S3IM_map = cw_ssim_map * Mask

    else:
        raise ValueError("Unknown SSIM variant passed.")

    # Compute final S3IM value
    if np.any(S3IM_map):
        S3IM_val = np.sum(S3IM_map) / np.sum(S3IM_map != 0)
    else:
        if increase_sensitivity == True:
            print('inc (2): ', sensitivity)
            increased_sensitivity=True
            return s3im(img1, img2, sensitivity=sensitivity+0.1, window_size=window_size, SSIM_variant=SSIM_variant,
                        increased_sensitivity=increased_sensitivity, increase_sensitivity=increase_sensitivity)
        else:
            return np.nan, 0, False

    return S3IM_val, Mask, S3IM_map

def msssim(img1, img2):
    ref = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0).float()   # [batch, channel, height, width]
    distorted = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).float()

    # Move to GPU if available
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ref = ref.to(device)
    distorted = distorted.to(device)

    # Calculate MS-SSIM
    msssim_value = np.float64(ms_ssim(ref, distorted, data_range=1.0, size_average=False).item())

    return msssim_value


# ms s3im
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def mss3im_ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None, mask=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = window_size // 2 
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if mask is not None:
        ssim_map = ssim_map * mask
        cs = cs * mask
        denom = torch.sum(mask)
        if denom > 0:
            ssim_score = torch.sum(ssim_map) / denom
            cs_score = torch.sum(cs) / denom
        else:
            ssim_score, cs_score = torch.tensor(0.0), torch.tensor(0.0)
    else:
        if size_average:
            cs_score = cs.mean()
            ssim_score = ssim_map.mean()
        else:
            cs_score = cs.mean(1).mean(1).mean(1)
            ssim_score = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ssim_score, cs_score, ssim_map
    return ssim_score

def run_mss3im(img1, img2, mask_np, window_size=11, size_average=True, val_range=None, roi=True):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []

    # Original mask (keep fixed, always resize based on current image shape)
    original_mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float().to(device)
    maps = []
    _, _, original_h, original_w = img1.shape

    for _ in range(levels):
        _, _, h, w = img1.shape

        if roi:
            # Resize original mask to match current image shape
            resized_mask = F.interpolate(original_mask, size=(h, w), mode='nearest')
        else:
            resized_mask = torch.ones((1, 1, h, w), device=device)

        sim, cs, ssim_map = mss3im_ssim(
            img1, img2,
            window_size=window_size,
            size_average=size_average,
            full=True,
            val_range=val_range,
            mask=resized_mask,
        )

        ssims.append(sim)
        mcs.append(cs)
        upsampled_map = F.interpolate(ssim_map, size=(original_h, original_w), mode='bilinear', align_corners=False)
        maps.append(upsampled_map)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    weight_tensor = weights.view(-1, 1, 1, 1)
    map_stack = torch.stack(maps)  # shape: (5, 1, 1, H, W)
    weight_tensor = weights.view(-1, 1, 1, 1, 1)  # shape: (5, 1, 1, 1, 1)
    weighted_map = torch.prod(map_stack ** weight_tensor, dim=0) 
    np_map = weighted_map.squeeze().detach().cpu().numpy().astype(np.float32)

    output = torch.prod(mcs[:-1] ** weights[:-1]) * (ssims[-1] ** weights[-1])
    return output, np_map, maps

def mss3im(img1, img2, roi=True):
    ref = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0).float()
    distorted = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).float()

    # Move to GPU if available
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ref = ref.to(device)
    distorted = distorted.to(device)

    np_mask = calculate_combined_mask(img1, img2)

    # Calculate MS-SSIM
    return  run_mss3im(ref, distorted, np_mask, val_range=1.0, roi=roi)


import numpy as np
from skimage.metrics import (
    structural_similarity,
)


def evaluate_metrics_on_distortions(
    imgs_ref,
    distort_fn,
    levels,
    increase_sensitivity= False,
):
    """
    Evaluate image-quality metrics over a set of distortions.

    Returns
    -------
    avg_scores
        Average score across reference images for each level.

    mean_taus
        Mean Kendall's tau per metric.

    mean_ps
        Mean Kendall p-value per metric.

    scores
        Dictionary:
            scores[metric][level_index] = [score_per_image, ...]

    sparsities
        Dictionary:
            sparsities[metric][level_index] = [sparsity_per_image, ...]

    df_results
        Tidy DataFrame containing, for every reference image,
        distortion level, metric, score, and sparsity.
    """

    # ========================================================
    # Existing output structures
    # ========================================================

    avg_scores = {
        "ssim": [],
        "wssim": [],
        "msssim": [],
        "s3im": [],
        "mss3im": [],
        "ws3im": [],
        "mse": [],
        "psnr": [],
        "snr": [],
    }

    scores = {
        "ssim": {},
        "wssim": {},
        "msssim": {},
        "s3im": {},
        "mss3im": {},
        "ws3im": {},
        "mse": {},
        "psnr": {},
        "snr": {},
    }

    sparsities = {
        "ssim": {},
        "wssim": {},
        "msssim": {},
        "s3im": {},
        "mss3im": {},
        "ws3im": {},
    }

    increases_sensitivities = {
        "s3im": {level: 0 for level in levels},
        "mss3im": {level: 0 for level in levels},
        "ws3im": {level: 0 for level in levels},
    }

    # Per-image scores for Kendall's tau
    scores_per_image = {
        idx: {
            k: []
            for k in avg_scores
            if k not in ("mse", "psnr", "snr")
        }
        for idx in range(len(imgs_ref))
    }

    # ========================================================
    # NEW: rows for the DataFrame
    # ========================================================

    df_rows = []

    # ========================================================
    # Iterate over distortion levels
    # ========================================================

    for i, level in enumerate(levels):

        ssim_scores = []
        wssim_scores = []
        msssim_scores = []

        s3im_scores = []
        mss3im_scores = []
        ws3im_scores = []

        mse_scores = []
        psnr_scores = []
        snr_scores = []

        ssim_sparsities = []
        wssim_sparsities = []
        msssim_sparsities = []

        s3im_sparsities = []
        mss3im_sparsities = []
        ws3im_sparsities = []

        # ====================================================
        # Iterate over reference images
        # ====================================================

        for idx, ref_img in enumerate(imgs_ref):

            # ------------------------------------------------
            # Create distorted image
            # ------------------------------------------------

            if distort_fn == "drop":

                distorted_img = drop_pixels(
                    ref_img,
                    level,
                    seed=42,
                )

            elif distort_fn == "noise":

                distorted_img = random_noise(
                    ref_img,
                    mode="gaussian",
                    var=level,
                    clip=True,
                    rng=42,
                )

            elif distort_fn == "blur":

                distorted_img = gaussian_filter(
                    ref_img,
                    sigma=level,
                )

            elif distort_fn == "noise inside":

                roi_mask = calculate_single_mask(
                    ref_img
                )

                noisy_img = random_noise(
                    ref_img,
                    mode="gaussian",
                    var=level,
                    clip=True,
                    rng=42,
                )

                distorted_img = (
                    ref_img * (1 - roi_mask)
                    + noisy_img * roi_mask
                )

            elif distort_fn == "noise outside":

                roi_mask = calculate_single_mask(
                    ref_img
                )

                noisy_img = random_noise(
                    ref_img,
                    mode="gaussian",
                    var=level,
                    clip=True,
                    rng=42,
                )

                distorted_img = (
                    ref_img * roi_mask
                    + noisy_img * (1 - roi_mask)
                )

            elif distort_fn == "blur inside":

                roi_mask = calculate_single_mask(
                    ref_img
                )

                blurred_img = gaussian_filter(
                    ref_img,
                    sigma=level,
                )

                distorted_img = (
                    ref_img * (1 - roi_mask)
                    + blurred_img * roi_mask
                )

            elif distort_fn == "blur outside":

                roi_mask = calculate_single_mask(
                    ref_img
                )

                blurred_img = gaussian_filter(
                    ref_img,
                    sigma=level,
                )

                distorted_img = (
                    ref_img * roi_mask
                    + blurred_img * (1 - roi_mask)
                )

            elif distort_fn == "shift":

                distorted_img = shift(
                    ref_img,
                    shift=(1.5, 0.5),
                    mode="reflect",
                )

            elif distort_fn == "block":

                roi_mask = calculate_single_mask(
                    ref_img
                )

                distorted_img, _ = mask_square_over_roi(
                    ref_img,
                    roi_mask,
                    level,
                )

            else:

                raise ValueError(
                    f"Unknown distortion type: {distort_fn}"
                )

            # ------------------------------------------------
            # Calculate metrics
            # ------------------------------------------------

            ssim_score = structural_similarity(
                ref_img,
                distorted_img,
                data_range=1.0,
            )

            wssim_score = np.average(
                cw_ssim(
                    ref_img,
                    distorted_img,
                )
            )

            msssim_score = msssim(
                ref_img,
                distorted_img,
            )

            mss3im_score, _, increases_mss3im = mss3im(
                ref_img,
                distorted_img,
            )

            s3im_score, _, increases_s3im = s3im(
                ref_img,
                distorted_img,
                increase_sensitivity=increase_sensitivity,
            )

            ws3im_score, _, increases_ws3im = s3im(
                ref_img,
                distorted_img,
                SSIM_variant=cw_ssim,
                increase_sensitivity=increase_sensitivity,
            )

            mse_score = mean_squared_error(
                ref_img,
                distorted_img,
            )

            psnr_score = peak_signal_noise_ratio(
                ref_img,
                distorted_img,
                data_range=1.0,
            )

            snr_score = snr_reference(
                ref_img,
                distorted_img,
            )


            # ------------------------------------------------
            # Calculate sparsity
            # ------------------------------------------------

            roi_mask = calculate_combined_mask(
                ref_img,
                distorted_img,
            )

            sparsity = roi_mask.mean()

            # ------------------------------------------------
            # Store existing score lists
            # ------------------------------------------------

            ssim_scores.append(ssim_score)
            wssim_scores.append(wssim_score)
            msssim_scores.append(msssim_score)

            s3im_scores.append(s3im_score)
            mss3im_scores.append(mss3im_score)
            ws3im_scores.append(ws3im_score)

            mse_scores.append(mse_score)
            psnr_scores.append(psnr_score)
            snr_scores.append(snr_score)

            # ------------------------------------------------
            # Store sparsities
            # ------------------------------------------------

            ssim_sparsities.append(sparsity)
            wssim_sparsities.append(sparsity)
            msssim_sparsities.append(sparsity)

            s3im_sparsities.append(sparsity)
            mss3im_sparsities.append(sparsity)
            ws3im_sparsities.append(sparsity)

            # ------------------------------------------------
            # Store per-image scores for Kendall tau
            # ------------------------------------------------

            scores_per_image[idx]["ssim"].append(
                ssim_score
            )

            scores_per_image[idx]["wssim"].append(
                wssim_score
            )

            scores_per_image[idx]["msssim"].append(
                msssim_score
            )

            scores_per_image[idx]["mss3im"].append(
                mss3im_score
            )

            scores_per_image[idx]["s3im"].append(
                s3im_score
            )

            scores_per_image[idx]["ws3im"].append(
                ws3im_score
            )

            # =================================================
            # NEW: Add rows to DataFrame
            # =================================================

            image_metrics = {
                "ssim": (
                    ssim_score,
                    sparsity,
                ),
                "wssim": (
                    wssim_score,
                    sparsity,
                ),
                "msssim": (
                    msssim_score,
                    sparsity,
                ),
                "s3im": (
                    s3im_score,
                    sparsity,
                ),
                "mss3im": (
                    mss3im_score,
                    sparsity,
                ),
                "ws3im": (
                    ws3im_score,
                    sparsity,
                ),
                "mse": (
                    mse_score,
                    np.nan,
                ),
                "psnr": (
                    psnr_score,
                    np.nan,
                ),
                "snr": (
                    snr_score,
                    np.nan,
                ),
            }

            for metric_name, (
                score,
                metric_sparsity,
            ) in image_metrics.items():

                df_rows.append(
                    {
                        "distortion_type": distort_fn,
                        "reference_image_idx": idx,
                        "distortion_level": level,
                        "level_idx": i,
                        "metric_variant": metric_name,
                        "score": score,
                        "sparsity": metric_sparsity,
                    }
                )

        # ====================================================
        # Store averages
        # ====================================================

        avg_scores["ssim"].append(
            np.nanmean(ssim_scores)
        )

        avg_scores["wssim"].append(
            np.nanmean(wssim_scores)
        )

        avg_scores["msssim"].append(
            np.nanmean(msssim_scores)
        )

        avg_scores["mss3im"].append(
            np.nanmean(mss3im_scores)
        )

        avg_scores["s3im"].append(
            np.nanmean(s3im_scores)
        )

        avg_scores["ws3im"].append(
            np.nanmean(ws3im_scores)
        )

        avg_scores["mse"].append(
            np.nanmean(mse_scores)
        )

        avg_scores["psnr"].append(
            np.nanmean(psnr_scores)
        )

        # FIX: your original code accidentally used psnr_scores here
        avg_scores["snr"].append(
            np.nanmean(snr_scores)
        )

        # ====================================================
        # Store per-level scores
        # ====================================================

        scores["ssim"][i] = ssim_scores
        scores["s3im"][i] = s3im_scores
        scores["wssim"][i] = wssim_scores
        scores["ws3im"][i] = ws3im_scores
        scores["msssim"][i] = msssim_scores
        scores["mss3im"][i] = mss3im_scores

        # ====================================================
        # Store per-level sparsities
        # ====================================================

        sparsities["ssim"][i] = ssim_sparsities
        sparsities["s3im"][i] = s3im_sparsities
        sparsities["wssim"][i] = wssim_sparsities
        sparsities["ws3im"][i] = ws3im_sparsities
        sparsities["msssim"][i] = msssim_sparsities
        sparsities["mss3im"][i] = mss3im_sparsities

    # ========================================================
    # Kendall's tau
    # ========================================================

    mean_taus = {}
    mean_ps = {}

    for metric in scores_per_image[0].keys():

        taus = []
        ps = []

        for idx in scores_per_image:

            tau, p = kendalltau(
                levels,
                scores_per_image[idx][metric],
            )

            taus.append(tau)
            ps.append(p)

        mean_taus[metric] = np.nanmean(taus)
        mean_ps[metric] = np.nanmean(ps)

    # ========================================================
    # Create DataFrame
    # ========================================================

    df_results = pd.DataFrame(df_rows)

    # ========================================================
    # Return everything
    # ========================================================

    return (
        avg_scores,
        mean_taus,
        mean_ps,
        scores,
        sparsities,
        df_results,
    )


def evaluate_metrics_on_pa_distortions(
    pa_from_base,
    levels=[1, 7, 6, 5, 4, 3, 2],
    increase_sensitivity=False,
):
    """
    Evaluate metrics on pre-computed frame-averaged (PA) images.

    Returns
    -------
    avg_scores
    mean_taus
    mean_ps
    scores
    sparsities
    df_results

    df_results contains one row per:

        reference image × PA level × metric

    with the corresponding score and sparsity.
    """

    # ========================================================
    # 1. Average scores storage
    # ========================================================

    avg_scores = {
        "ssim": [],
        "wssim": [],
        "msssim": [],
        "s3im": [],
        "mss3im": [],
        "ws3im": [],
        "mse": [],
        "psnr": [],
        "snr": [],
    }

    # ========================================================
    # 2. Detailed per-level scores
    # ========================================================

    scores = {
        "ssim": {},
        "wssim": {},
        "msssim": {},
        "s3im": {},
        "mss3im": {},
        "ws3im": {},
        "mse": {},
        "psnr": {},
        "snr": {},
    }

    # ========================================================
    # 3. Sparsity storage
    # ========================================================

    sparsities = {
        "ssim": {},
        "wssim": {},
        "msssim": {},
        "s3im": {},
        "mss3im": {},
        "ws3im": {},
    }

    # ========================================================
    # Image ordering
    # ========================================================

    image_keys = list(pa_from_base.keys())

    # ========================================================
    # Per-image metric progressions for Kendall's tau
    # ========================================================

    scores_per_image = {
        idx: {
            k: []
            for k in avg_scores
            if k not in ("mse", "psnr", "snr")
        }
        for idx in range(len(image_keys))
    }

    # ========================================================
    # NEW: rows for tidy DataFrame
    # ========================================================

    df_rows = []

    # ========================================================
    # 4. Process PA levels
    # ========================================================

    for i, level in enumerate(levels):

        ssim_scores = []
        wssim_scores = []
        msssim_scores = []

        s3im_scores = []
        mss3im_scores = []
        ws3im_scores = []

        mse_scores = []
        psnr_scores = []
        snr_scores = []

        ssim_sparsities = []
        wssim_sparsities = []
        msssim_sparsities = []

        s3im_sparsities = []
        mss3im_sparsities = []
        ws3im_sparsities = []

        # ====================================================
        # Process each reference image
        # ====================================================

        for idx, key in enumerate(image_keys):

            ref_img = pa_from_base[key]["PA1"]

            distorted_img = pa_from_base[key][f"PA{level}"]

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            ssim_score = structural_similarity(
                ref_img,
                distorted_img,
                data_range=1.0,
            )

            wssim_score = np.average(
                cw_ssim(
                    ref_img,
                    distorted_img,
                )
            )

            msssim_score = msssim(
                ref_img,
                distorted_img,
            )

            mss3im_score, _, _ = mss3im(
                ref_img,
                distorted_img,
            )

            s3im_score, _, _ = s3im(
                ref_img,
                distorted_img,
                increase_sensitivity=increase_sensitivity,
            )

            ws3im_score, _, _ = s3im(
                ref_img,
                distorted_img,
                SSIM_variant=cw_ssim,
                increase_sensitivity=increase_sensitivity,
            )

            mse_score = mean_squared_error(
                ref_img,
                distorted_img,
            )

            psnr_score = peak_signal_noise_ratio(
                ref_img,
                distorted_img,
                data_range=1.0,
            )

            snr_score = snr_reference(
                ref_img,
                distorted_img,
            )

            # ------------------------------------------------
            # Store scores
            # ------------------------------------------------

            ssim_scores.append(ssim_score)
            wssim_scores.append(wssim_score)
            msssim_scores.append(msssim_score)

            s3im_scores.append(s3im_score)
            mss3im_scores.append(mss3im_score)
            ws3im_scores.append(ws3im_score)

            mse_scores.append(mse_score)
            psnr_scores.append(psnr_score)
            snr_scores.append(snr_score)

            # ------------------------------------------------
            # Combined mask sparsity
            # ------------------------------------------------

            roi_mask = calculate_combined_mask(
                ref_img,
                distorted_img,
            )

            sparsity = roi_mask.mean()

            ssim_sparsities.append(sparsity)
            wssim_sparsities.append(sparsity)
            msssim_sparsities.append(sparsity)

            s3im_sparsities.append(sparsity)
            mss3im_sparsities.append(sparsity)
            ws3im_sparsities.append(sparsity)

            # ------------------------------------------------
            # Per-image metric progression
            # ------------------------------------------------

            scores_per_image[idx]["ssim"].append(
                ssim_score
            )

            scores_per_image[idx]["wssim"].append(
                wssim_score
            )

            scores_per_image[idx]["msssim"].append(
                msssim_score
            )

            scores_per_image[idx]["mss3im"].append(
                mss3im_score
            )

            scores_per_image[idx]["s3im"].append(
                s3im_score
            )

            scores_per_image[idx]["ws3im"].append(
                ws3im_score
            )

            # =================================================
            # NEW: DataFrame rows
            # =================================================

            image_metrics = {
                "ssim": (
                    ssim_score,
                    sparsity,
                ),
                "wssim": (
                    wssim_score,
                    sparsity,
                ),
                "msssim": (
                    msssim_score,
                    sparsity,
                ),
                "s3im": (
                    s3im_score,
                    sparsity,
                ),
                "mss3im": (
                    mss3im_score,
                    sparsity,
                ),
                "ws3im": (
                    ws3im_score,
                    sparsity,
                ),
                "mse": (
                    mse_score,
                    np.nan,
                ),
                "psnr": (
                    psnr_score,
                    np.nan,
                ),
                "snr": (
                    snr_score,
                    np.nan,
                ),
            }

            for metric_name, (
                score,
                metric_sparsity,
            ) in image_metrics.items():

                df_rows.append(
                    {
                        "distortion_type": "pa",
                        "reference_image_idx": idx,
                        "reference_image_key": key,
                        "distortion_level": level,
                        "level_idx": i,
                        "metric_variant": metric_name,
                        "score": score,
                        "sparsity": metric_sparsity,
                    }
                )

        # ====================================================
        # 5. Average scores
        # ====================================================

        avg_scores["ssim"].append(
            np.nanmean(ssim_scores)
        )

        avg_scores["wssim"].append(
            np.nanmean(wssim_scores)
        )

        avg_scores["msssim"].append(
            np.nanmean(msssim_scores)
        )

        avg_scores["mss3im"].append(
            np.nanmean(mss3im_scores)
        )

        avg_scores["s3im"].append(
            np.nanmean(s3im_scores)
        )

        avg_scores["ws3im"].append(
            np.nanmean(ws3im_scores)
        )

        avg_scores["mse"].append(
            np.nanmean(mse_scores)
        )

        avg_scores["psnr"].append(
            np.nanmean(psnr_scores)
        )

        avg_scores["snr"].append(
            np.nanmean(snr_scores)
        )

        # ====================================================
        # 6. Store scores by level index
        # ====================================================

        scores["ssim"][i] = ssim_scores
        scores["s3im"][i] = s3im_scores
        scores["wssim"][i] = wssim_scores
        scores["ws3im"][i] = ws3im_scores
        scores["msssim"][i] = msssim_scores
        scores["mss3im"][i] = mss3im_scores

        scores["mse"][i] = mse_scores
        scores["psnr"][i] = psnr_scores
        scores["snr"][i] = snr_scores

        # ====================================================
        # 7. Store sparsities by level index
        # ====================================================

        sparsities["ssim"][i] = ssim_sparsities
        sparsities["s3im"][i] = s3im_sparsities
        sparsities["wssim"][i] = wssim_sparsities
        sparsities["ws3im"][i] = ws3im_sparsities
        sparsities["msssim"][i] = msssim_sparsities
        sparsities["mss3im"][i] = mss3im_sparsities

    # ========================================================
    # 8. Kendall's tau
    # ========================================================

    mean_taus = {}
    mean_ps = {}

    for metric in scores_per_image[0].keys():

        taus = []
        ps = []

        for idx in scores_per_image:

            tau, p = kendalltau(
                levels,
                scores_per_image[idx][metric],
            )

            taus.append(tau)
            ps.append(p)

        mean_taus[metric] = np.nanmean(taus)
        mean_ps[metric] = np.nanmean(ps)

    # ========================================================
    # 9. Create tidy DataFrame
    # ========================================================

    df_results = pd.DataFrame(df_rows)

    # ========================================================
    # 10. Return original outputs + DataFrame
    # ========================================================

    return (
        avg_scores,
        mean_taus,
        mean_ps,
        scores,
        sparsities,
        df_results,
    )
