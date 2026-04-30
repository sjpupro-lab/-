"""
Row-wise Residual VAE-like Image Generation Skeleton

Core idea:
- Create a low-resolution latent base.
- Upscale it to a base image.
- Learn residual = original - base.
- Treat each image row as a signal.
- Learn row-wise residual basis.
- Generate new images by mixing latent bases and residual row coefficients.

This is a compact reference implementation based on the experiments.
"""

import numpy as np
from PIL import Image


def to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB")).astype(np.float32) / 255.0


def to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGB")


def make_latent_and_base(img: Image.Image, latent_size: int, out_size: int):
    latent = img.resize((latent_size, latent_size), Image.Resampling.LANCZOS)
    base = latent.resize((out_size, out_size), Image.Resampling.LANCZOS)
    return latent, base


def train_row_residual_basis(images, latent_size=32, out_size=128, k=24):
    refs, latents, bases, residuals = [], [], [], []

    for img in images:
        img = img.convert("RGB").resize((out_size, out_size), Image.Resampling.LANCZOS)
        latent, base = make_latent_and_base(img, latent_size, out_size)

        ref_arr = to_array(img)
        latent_arr = to_array(latent)
        base_arr = to_array(base)

        refs.append(ref_arr)
        latents.append(latent_arr)
        bases.append(base_arr)
        residuals.append(ref_arr - base_arr)

    refs = np.stack(refs)
    latents = np.stack(latents)
    bases = np.stack(bases)
    residuals = np.stack(residuals)

    n, h, w, c = residuals.shape
    rows = residuals.reshape(n * h, w * c)

    mean_row = rows.mean(axis=0, keepdims=True)
    centered = rows - mean_row

    # Row-wise basis learning
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:k]

    coeffs = (centered @ basis.T).reshape(n, h, k)

    return {
        "refs": refs,
        "latents": latents,
        "bases": bases,
        "residuals": residuals,
        "mean_row": mean_row,
        "basis": basis,
        "coeffs": coeffs,
        "latent_size": latent_size,
        "out_size": out_size,
        "k": k,
    }


def decode(base_arr, coeff_arr, model):
    h = model["out_size"]
    w = model["out_size"]
    k = model["k"]

    residual_rows = coeff_arr @ model["basis"] + model["mean_row"]
    residual = residual_rows.reshape(h, w, 3)

    return np.clip(base_arr + residual, 0, 1)


def generate_mixed(model, parent_ids, weights=None, noise_strength=0.2):
    latents = model["latents"]
    coeffs = model["coeffs"]
    out_size = model["out_size"]

    parent_ids = np.array(parent_ids, dtype=int)

    if weights is None:
        weights = np.ones(len(parent_ids), dtype=np.float32) / len(parent_ids)
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / weights.sum()

    latent_mix = np.sum(latents[parent_ids] * weights[:, None, None, None], axis=0)
    latent_img = to_image(latent_mix)
    base_img = latent_img.resize((out_size, out_size), Image.Resampling.LANCZOS)
    base_arr = to_array(base_img)

    coeff_mix = np.sum(coeffs[parent_ids] * weights[:, None, None], axis=0)

    # Structured residual/noise: difference between two learned coefficient fields
    a, b = np.random.choice(len(latents), 2, replace=False)
    coeff_noise = coeffs[a] - coeffs[b]
    coeff_new = coeff_mix + noise_strength * coeff_noise

    # Simple row smoothing
    for i in range(model["k"]):
        coeff_new[:, i] = np.convolve(coeff_new[:, i], np.ones(3) / 3, mode="same")

    return decode(base_arr, coeff_new, model)


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def psnr(a, b):
    m = mse(a, b)
    return float(10 * np.log10(1.0 / m)) if m > 0 else float("inf")
