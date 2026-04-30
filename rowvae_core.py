
"""
Row-wise Residual VAE-like Generator Core

This is a compact implementation skeleton based on the experiments:
- low-resolution latent base
- residual = original - upscaled latent
- row-wise residual basis learning
- generation via latent mixing + residual coefficient mixing/noise
"""

import numpy as np
from PIL import Image

def to_array(img):
    return np.asarray(img.convert("RGB")).astype(np.float32) / 255.0

def to_image(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGB")

def make_base(img, latent_size=32, out_size=128):
    latent = img.resize((latent_size, latent_size), Image.Resampling.LANCZOS)
    base = latent.resize((out_size, out_size), Image.Resampling.LANCZOS)
    return latent, base

def train_row_residual_basis(images, latent_size=32, out_size=128, k=24):
    refs = np.stack([to_array(im.resize((out_size, out_size), Image.Resampling.LANCZOS)) for im in images])
    latents = []
    bases = []
    for im in images:
        latent, base = make_base(im, latent_size, out_size)
        latents.append(to_array(latent))
        bases.append(to_array(base))
    latents = np.stack(latents)
    bases = np.stack(bases)
    residuals = refs - bases

    n, h, w, c = residuals.shape
    rows = residuals.reshape(n * h, w * c)
    mean = rows.mean(axis=0, keepdims=True)
    centered = rows - mean

    # row-wise basis
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:k]

    coeffs = (centered @ basis.T).reshape(n, h, k)

    return {
        "latents": latents,
        "bases": bases,
        "refs": refs,
        "mean": mean,
        "basis": basis,
        "coeffs": coeffs,
        "latent_size": latent_size,
        "out_size": out_size,
        "k": k,
    }

def decode(base_arr, coeff_arr, mean, basis):
    h, w, c = base_arr.shape
    residual_rows = coeff_arr @ basis + mean
    residual = residual_rows.reshape(h, w, c)
    return np.clip(base_arr + residual, 0, 1)

def mix_generate(model, parent_ids, weights, noise_pair=None, noise_strength=0.2, smooth=True):
    latents = model["latents"]
    coeffs = model["coeffs"]
    mean = model["mean"]
    basis = model["basis"]
    out_size = model["out_size"]

    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / weights.sum()

    latent_mix = np.sum(latents[parent_ids] * weights[:, None, None, None], axis=0)
    latent_img = to_image(latent_mix)
    base_mix = to_array(latent_img.resize((out_size, out_size), Image.Resampling.LANCZOS))

    coeff_mix = np.sum(coeffs[parent_ids] * weights[:, None, None], axis=0)

    if noise_pair is not None:
        a, b = noise_pair
        coeff_mix = coeff_mix + noise_strength * (coeffs[a] - coeffs[b])

    if smooth:
        for k in range(coeff_mix.shape[1]):
            coeff_mix[:, k] = np.convolve(coeff_mix[:, k], np.ones(3) / 3, mode="same")

    return decode(base_mix, coeff_mix, mean, basis)
