"""
Feature extraction for raw MRI volumes (Analyze .hdr/.img or NIfTI .nii/.nii.gz).

This mirrors the per-slice feature pipeline from Model1.ipynb
(extract_slice_features / extract_features_from_volume) so that a batch
of raw scans can be scored with the same XGBoost model trained on the
pre-computed feature CSV.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import cv2
from scipy.stats import entropy
from scipy.ndimage import sobel, laplace, gaussian_filter
from skimage.feature import graycomatrix, graycoprops

ALL_FEATURE_COLS = [
    "Mean", "Std", "Variance", "SNR", "Entropy", "Sharpness",
    "LaplacianVariance", "GradientMagnitude", "GLCMContrast",
    "GLCMCorrelation", "GLCMEnergy", "GLCMHomogeneity",
    "BiasQuadrantRange", "BiasGradientMagnitude",
]

BIAS_FEATURE_COLS = ["BiasQuadrantRange", "BiasGradientMagnitude"]


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    volume = volume - np.min(volume)
    volume = volume / (np.max(volume) + 1e-8)
    return volume


def extract_bias_features(slice_img: np.ndarray) -> dict:
    """Features that specifically target smooth, low-frequency bias-field
    intensity variation (RF coil inhomogeneity etc.) — the kind of
    artifact that whole-slice statistics like Mean/Entropy/GLCM wash out,
    since they average over the whole image rather than looking at
    *where* the intensity trend sits spatially.

    - BiasQuadrantRange: split the foreground into a 2x2 grid, take the
      spread between quadrant means, normalized by overall mean. A bias
      field skews intensity systematically across the image, producing a
      larger spread than a clean/uniform scan.
    - BiasGradientMagnitude: heavily blur the image (to isolate the
      low-frequency component), fit a linear plane a*x + b*y + c to the
      blurred foreground intensities by least squares, and report the
      magnitude of the (a, b) gradient. A directional bias field produces
      a consistent linear trend; a clean scan's blurred intensity should
      be comparatively flat.
    """
    h, w = slice_img.shape
    mask = slice_img > (0.05 * slice_img.max() + 1e-8)
    if mask.sum() < 16:
        return {"BiasQuadrantRange": 0.0, "BiasGradientMagnitude": 0.0}

    fg_mean = float(slice_img[mask].mean()) + 1e-8

    quads = [slice_img[:h // 2, :w // 2], slice_img[:h // 2, w // 2:],
             slice_img[h // 2:, :w // 2], slice_img[h // 2:, w // 2:]]
    qmasks = [mask[:h // 2, :w // 2], mask[:h // 2, w // 2:],
              mask[h // 2:, :w // 2], mask[h // 2:, w // 2:]]
    qmeans = np.array([q[m].mean() if m.sum() > 0 else fg_mean
                        for q, m in zip(quads, qmasks)])
    quadrant_range = float((qmeans.max() - qmeans.min()) / fg_mean)

    blurred = gaussian_filter(slice_img, sigma=max(h, w) / 8)
    yy, xx = np.mgrid[0:h, 0:w]
    ys, xs, vs = yy[mask].astype(np.float64), xx[mask].astype(np.float64), blurred[mask].astype(np.float64)
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    coef, *_ = np.linalg.lstsq(A, vs, rcond=None)
    grad_mag = float(np.hypot(coef[0] * w, coef[1] * h) / (vs.mean() + 1e-8))

    return {"BiasQuadrantRange": quadrant_range, "BiasGradientMagnitude": grad_mag}


def extract_slice_features(slice_img: np.ndarray) -> dict:
    slice_img = np.nan_to_num(slice_img).astype(np.float32)
    slice_img = slice_img - slice_img.min()
    slice_img = slice_img / (slice_img.max() + 1e-8)
    img8 = (slice_img * 255).astype(np.uint8)

    mean = float(np.mean(slice_img))
    std = float(np.std(slice_img))
    variance = float(np.var(slice_img))
    snr = mean / (std + 1e-8)

    hist = cv2.calcHist([img8], [0], None, [256], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-8)
    ent = float(entropy(hist))

    gx = sobel(slice_img, axis=0)
    gy = sobel(slice_img, axis=1)
    gradient = np.sqrt(gx ** 2 + gy ** 2)
    sharpness = float(np.mean(gradient))

    lap = laplace(slice_img)
    lap_var = float(np.var(lap))

    grad = float(np.mean(gradient))

    glcm = graycomatrix(img8, [1], [0], 256, symmetric=True, normed=True)
    contrast = float(graycoprops(glcm, "contrast")[0, 0])
    correlation = float(graycoprops(glcm, "correlation")[0, 0])
    energy = float(graycoprops(glcm, "energy")[0, 0])
    homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])

    feats = {
        "Mean": mean, "Std": std, "Variance": variance, "SNR": snr,
        "Entropy": ent, "Sharpness": sharpness, "LaplacianVariance": lap_var,
        "GradientMagnitude": grad, "GLCMContrast": contrast,
        "GLCMCorrelation": correlation, "GLCMEnergy": energy,
        "GLCMHomogeneity": homogeneity,
    }
    feats.update(extract_bias_features(slice_img))
    return feats


def extract_features_from_volume(volume: np.ndarray, max_slices: int | None = None) -> pd.DataFrame:
    """Per-axial-slice features. Optionally subsample slices (max_slices)
    for speed on large volumes — the notebook processes every slice, but
    a stride keeps interactive dashboards responsive on big batches.
    """
    n_slices = volume.shape[2]
    if max_slices and n_slices > max_slices:
        idx = np.linspace(0, n_slices - 1, max_slices).astype(int)
    else:
        idx = range(n_slices)

    rows = []
    for z in idx:
        slice_img = volume[:, :, z]
        if slice_img.max() - slice_img.min() < 1e-6:
            continue  # skip blank/empty slices
        rows.append(extract_slice_features(slice_img))
    return pd.DataFrame(rows)


def load_volume(hdr_path: str) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load an Analyze (.hdr/.img) or NIfTI volume, returning a squeezed
    3D array and the source image object (for shape/affine info)."""
    img = nib.load(hdr_path)
    volume = np.squeeze(np.asarray(img.get_fdata()))
    if volume.ndim > 3:
        # collapse any remaining trailing dims (e.g. time) by taking the first frame
        volume = volume[..., 0]
        while volume.ndim > 3:
            volume = volume[..., 0]
    return volume, img


def extract_scan_features(hdr_path: str, max_slices: int | None = 64) -> dict:
    """Load one volume and return a single aggregated feature row
    (mean across slices), matching `selected_features` used to train
    the XGBoost model plus the full feature set for display."""
    volume, img = load_volume(hdr_path)
    volume = normalize_volume(volume)
    slice_df = extract_features_from_volume(volume, max_slices=max_slices)
    if slice_df.empty:
        raise ValueError(f"No usable (non-blank) slices found in {hdr_path}")
    agg = slice_df.mean().to_dict()
    agg["_shape"] = tuple(volume.shape)
    agg["_n_slices_used"] = len(slice_df)
    agg["_n_slices_total"] = volume.shape[2]
    return agg


def find_hdr_img_pairs(root_dir: str) -> tuple[list[dict], list[str]]:
    """Walk a directory recursively and pair up .hdr/.img files that
    share the same base filename within the same folder. Standalone
    .nii / .nii.gz files (already self-contained) are also picked up
    as single-file "pairs".

    Returns (pairs, warnings) where pairs is a list of
    {"name": str, "hdr": path, "img": path_or_None} and warnings lists
    any unmatched .hdr or .img files found.
    """
    hdrs, imgs = {}, {}
    nifti_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            lower = fn.lower()
            full = os.path.join(dirpath, fn)

            if lower.endswith(".nii.gz"):
                nifti_files.append((fn[:-7], full))
                continue
            if lower.endswith(".nii"):
                nifti_files.append((fn[:-4], full))
                continue

            stem, ext = os.path.splitext(fn)
            ext = ext.lower()
            key = (dirpath, stem.lower())
            if ext == ".hdr":
                hdrs[key] = (stem, full)
            elif ext == ".img":
                imgs[key] = (stem, full)

    pairs, warnings = [], []

    for key, (stem, hdr_path) in hdrs.items():
        if key in imgs:
            pairs.append({"name": stem, "hdr": hdr_path, "img": imgs[key][1]})
        else:
            warnings.append(f"No matching .img for {hdr_path}")
    for key, (stem, img_path) in imgs.items():
        if key not in hdrs:
            warnings.append(f"No matching .hdr for {img_path}")

    for stem, path in nifti_files:
        pairs.append({"name": stem, "hdr": path, "img": None})

    pairs.sort(key=lambda p: p["name"])
    return pairs, warnings
