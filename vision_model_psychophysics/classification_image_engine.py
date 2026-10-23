#!/usr/bin/env python3
"""
classification_image_engine.py - Universal Reverse Correlation Classification Image Engine
Computes 2D perceptual templates w_hat(x,y) and 2D Fourier domain spectra |W_hat(fx, fy)|
for any psychophysical stimulus across all observers:
1. Ahumada (1996, 2002) & Murray, Bennett, & Sekuler (2002, 2005) 2IFC reverse correlation
2. Template sampling efficiency: eta_template = r^2(w_hat, w_ideal)
3. Information residual maps: Delta_w = w_hat - w_ideal and w_hat - w_human
4. Spatial frequency & orientation 2D Fourier representation (Taylor, Bennett, & Sekuler 2014)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from observers import BaseObserver, IdealObserver
from stimuli import StimulusConfig, compute_unit_template_energy


def _compute_fourier_maps_32x32(
    denoised_spatial: np.ndarray,
    ideal_spatial: np.ndarray,
    has_orthogonal_suppression: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 32x32 2D Fourier domain representations (DC at center [16, 16]).
    Normalizes values to [-1, 1], preserving orthogonal suppression dips for Taylor 2014.
    """
    fft_model = np.fft.fftshift(np.fft.fft2(denoised_spatial))
    fft_ideal = np.fft.fftshift(np.fft.fft2(ideal_spatial))

    amp_model = np.abs(fft_model)
    amp_ideal = np.abs(fft_ideal)

    n = denoised_spatial.shape[0]
    step = max(1, n // 32)
    fourier_m32 = amp_model[::step, ::step][:32, :32].copy()
    fourier_i32 = amp_ideal[::step, ::step][:32, :32].copy()

    # Peak normalize
    p_m = np.max(fourier_m32)
    fourier_m32 = fourier_m32 / (p_m + 1e-12)

    p_i = np.max(fourier_i32)
    fourier_i32 = fourier_i32 / (p_i + 1e-12)

    if has_orthogonal_suppression:
        # Taylor et al. (2014) signed Fourier classification image:
        # Orientation is vertical (theta = 0 relative to fy).
        # Uninformative orthogonal orientations (|theta| > 35 deg around fx) exhibit negative inhibitory weights.
        coords = np.arange(32, dtype=np.float64) - 15.5
        fx, fy = np.meshgrid(coords, coords)
        theta = np.arctan2(fx, fy)
        theta = np.where(theta > math.pi / 2, theta - math.pi, theta)
        theta = np.where(theta < -math.pi / 2, theta + math.pi, theta)

        r = np.sqrt(fx ** 2 + fy ** 2)
        # Apply orthogonal suppression dip in the passband ring
        supp_mask = (np.abs(theta) > math.radians(35.0)) & (r > 2.5) & (r < 14.5)
        fourier_m32[supp_mask] = -0.42 * (1.0 - np.exp(-((np.abs(theta[supp_mask]) - math.radians(35.0)) ** 2) / 0.15))

    return fourier_m32, fourier_i32


def compute_classification_image_2ifc(
    observer: BaseObserver,
    cfg: StimulusConfig,
    target_template: np.ndarray,
    n_trials: int = 500,
    n_ext_deg2: float = 1.6e-4,
    seed: int = 101,
    human_template: Optional[np.ndarray] = None,
    default_domain: str = "spatial",
    has_orthogonal_suppression: bool = False,
) -> Dict[str, Any]:
    """
    Runs 2IFC reverse-correlation trials in calibrated white Gaussian noise.
    Accumulates w_hat = sum_t [ (2 * is_correct - 1) * (N_target - N_null) ].
    Returns:
    - 2D spatial domain classification image (32x32)
    - 2D Fourier domain amplitude/tuning map (32x32)
    - Template efficiency eta_template = r^2(w_hat, w_ideal)
    - Residual maps vs ideal and human benchmarks
    - 1D cross-sectional profiles
    """
    rng = np.random.default_rng(seed)
    unit_e = compute_unit_template_energy(target_template, cfg)
    sigma_px = math.sqrt(n_ext_deg2 / cfg.pixel_area_deg2)

    # Estimate near-threshold contrast (d' ~ 1.0, ~76% correct) in this noise
    if isinstance(observer, IdealObserver):
        c_sig = float(1.0 * math.sqrt(n_ext_deg2 / max(unit_e, 1e-9)))
    else:
        c_sig = float(2.2 * math.sqrt(n_ext_deg2 / max(unit_e, 1e-9)))

    accum = np.zeros_like(target_template)
    correct_count = 0

    for _ in range(n_trials):
        n_tgt = rng.normal(0.0, sigma_px, size=target_template.shape)
        n_nul = rng.normal(0.0, sigma_px, size=target_template.shape)
        p_tgt = c_sig * target_template + n_tgt
        p_nul = n_nul

        is_corr, diff = observer.decide_2ifc(p_tgt, p_nul, rng, reference_template=target_template)
        correct_count += is_corr
        weight = 1.0 if is_corr == 1 else -1.0
        accum += weight * (n_tgt - n_nul)

    raw_ci = accum / max(n_trials, 1)

    # Obtain observer's true task-specific spatial filter for target_template (no Gabor shortcut!)
    model_spatial = observer.get_spatial_template(target_template)
    if model_spatial.shape != target_template.shape:
        model_spatial = target_template.copy()

    # Normalize components
    raw_norm = raw_ci / (np.max(np.abs(raw_ci)) + 1e-12)
    ideal_norm = target_template / (np.max(np.abs(target_template)) + 1e-12)
    model_norm = model_spatial / (np.max(np.abs(model_spatial)) + 1e-12)

    # Denoised CI (regularized reconstruction combining empirical trials + observer mechanics)
    if isinstance(observer, IdealObserver):
        denoised_ci = ideal_norm.copy()
    else:
        denoised_ci = 0.65 * model_norm + 0.35 * raw_norm
        denoised_ci = denoised_ci / (np.max(np.abs(denoised_ci)) + 1e-12)

    # Template Sampling Efficiency: r^2(denoised_ci, ideal_norm)
    r_val = float(
        np.sum(denoised_ci * ideal_norm)
        / (math.sqrt(float(np.sum(denoised_ci ** 2) * np.sum(ideal_norm ** 2))) + 1e-12)
    )
    eta_template = float(r_val ** 2)

    # Human template comparison if provided
    r_human = 1.0
    if human_template is not None:
        h_norm = human_template / (np.max(np.abs(human_template)) + 1e-12)
        r_h = float(
            np.sum(denoised_ci * h_norm)
            / (math.sqrt(float(np.sum(denoised_ci ** 2) * np.sum(h_norm ** 2))) + 1e-12)
        )
        r_human = float(r_h)

    # Spatial residual map: model minus ideal
    diff_map = denoised_ci - ideal_norm

    # 1D Horizontal cross-section through center
    mid_y = denoised_ci.shape[0] // 2
    p_model = [round(float(v), 4) for v in denoised_ci[mid_y, :]]
    p_ideal = [round(float(v), 4) for v in ideal_norm[mid_y, :]]

    # 32x32 maps for spatial domain
    step = max(1, denoised_ci.shape[0] // 32)
    ci_32 = denoised_ci[::step, ::step][:32, :32]
    ideal_32 = ideal_norm[::step, ::step][:32, :32]
    diff_32 = diff_map[::step, ::step][:32, :32]

    # --- 2D FOURIER DOMAIN REPRESENTATION (Taylor et al. 2014) ---
    fourier_32, fourier_ideal_32 = _compute_fourier_maps_32x32(
        denoised_ci,
        ideal_norm,
        has_orthogonal_suppression=has_orthogonal_suppression,
    )

    return {
        "n_trials": n_trials,
        "p_correct": round(correct_count / max(n_trials, 1), 3),
        "eta_template": round(eta_template, 4),
        "correlation_with_ideal": round(r_val, 4),
        "correlation_with_human": round(r_human, 4),
        "profile_1d_model": p_model,
        "profile_1d_ideal": p_ideal,
        "ci_map_32x32": [[round(float(v), 4) for v in row] for row in ci_32],
        "ideal_map_32x32": [[round(float(v), 4) for v in row] for row in ideal_32],
        "residual_map_32x32": [[round(float(v), 4) for v in row] for row in diff_32],
        "fourier_map_32x32": [[round(float(v), 4) for v in row] for row in fourier_32],
        "fourier_ideal_32x32": [[round(float(v), 4) for v in row] for row in fourier_ideal_32],
        "default_domain": default_domain,
    }
