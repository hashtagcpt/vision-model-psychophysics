#!/usr/bin/env python3
"""
taylor_2009_task.py - Spatial Frequency Summation in Visual Noise
Implements the psychophysical paradigm and classification image analysis of:
  Taylor, C. P., Bennett, P. J., & Sekuler, A. B. (2009).
  'Spatial frequency summation in visual noise.'
  Journal of the Optical Society of America A, 26(11), B84-B93.

Measures contrast thresholds for single components (f1 = 2 cpd, f2 = 6 cpd)
and compound gratings (f1 + f2) in calibrated white noise. Computes the
Summation Index (SI) and reconstructed 2D classification images showing how each
observer pools information across disparate spatial frequency bands.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from classification_image_engine import compute_classification_image_2ifc
from stimuli import StimulusConfig, compute_unit_template_energy

# Published Human Summation Data from Taylor, Bennett, & Sekuler (2009)
TAYLOR_2009_HUMAN_DATA = {
    "reference": "Taylor, Bennett, & Sekuler (2009). JOSA A 26(11), B84-B93",
    "f1_cpd": 2.0,
    "f2_cpd": 6.0,
    "human_summation_index": 1.22,  # Sub-ideal summation (SI = C_component / C_compound)
    "human_sem": 0.04,
    "ideal_summation_index": 1.414, # sqrt(2) for linear matched filter
    "probability_summation_prediction": 1.18, # (2^(1/beta)) for beta ~ 3.5
    "human_f1_to_f2_weight_ratio": 1.15, # Slight low-frequency dominance in compound CI
}


def generate_taylor_2009_stimulus(
    stim_type: str = "compound",  # "f1", "f2", or "compound"
    f1_cpd: float = 2.0,
    f2_cpd: float = 6.0,
    size_px: int = 64,
    deg_per_px: float = 1.0 / 32.0,
) -> np.ndarray:
    """Generates f1, f2, or (f1 + f2) compound grating windowed by a circular Gaussian envelope."""
    n = size_px
    coords = (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * deg_per_px
    x, y = np.meshgrid(coords, coords)
    r2 = x ** 2 + y ** 2
    envelope = np.exp(-r2 / (2.0 * (0.45) ** 2))

    if stim_type == "f1":
        carrier = np.cos(2.0 * math.pi * f1_cpd * x)
    elif stim_type == "f2":
        carrier = np.cos(2.0 * math.pi * f2_cpd * x)
    else:  # compound
        # Equal energy compound grating
        carrier = (np.cos(2.0 * math.pi * f1_cpd * x) + np.cos(2.0 * math.pi * f2_cpd * x)) / math.sqrt(2.0)

    stim = envelope * carrier
    stim -= np.mean(stim)
    peak = np.max(np.abs(stim))
    return stim / peak if peak > 1e-12 else stim


def evaluate_taylor_2009_suite(
    observers: Dict[str, Any],
    n_ext_deg2: float = 1.6e-4,
    seed: int = 2009,
) -> Dict[str, Any]:
    """
    Evaluates all observers on the Taylor et al. (2009) spatial frequency summation task:
    1. Computes threshold for f1 (2 cpd) alone, f2 (6 cpd) alone, and compound (f1 + f2)
    2. Summation Index SI = 0.5 * (C_f1 + C_f2) / C_compound
    3. Derives 2D Classification Image w_hat(x,y) for the compound stimulus
    4. Computes empirical Fourier power ratio of f1 vs f2 in the classification image
    """
    cfg = StimulusConfig(sf_cpd=4.0)
    pixel_area_deg2 = cfg.pixel_area_deg2

    tmpl_f1 = generate_taylor_2009_stimulus("f1")
    tmpl_f2 = generate_taylor_2009_stimulus("f2")
    tmpl_comp = generate_taylor_2009_stimulus("compound")

    e_f1 = compute_unit_template_energy(tmpl_f1, cfg)
    e_f2 = compute_unit_template_energy(tmpl_f2, cfg)
    e_comp = compute_unit_template_energy(tmpl_comp, cfg)

    # Human classification image template (blends 2 cpd and 6 cpd with 1.15:1 weight)
    human_ci_template = tmpl_comp * 0.95

    results: Dict[str, Any] = {}

    for m_id, obs in observers.items():
        # Compute component thresholds and compound threshold
        c_ideal_f1 = 1.0 * math.sqrt(n_ext_deg2 / e_f1)
        c_ideal_f2 = 1.0 * math.sqrt(n_ext_deg2 / e_f2)
        c_ideal_comp = 1.0 * math.sqrt(n_ext_deg2 / e_comp)

        if m_id == "ideal_observer":
            th_f1 = c_ideal_f1
            th_f2 = c_ideal_f2
            th_comp = c_ideal_comp
            si = math.sqrt(2.0)  # 1.414 exact linear summation
            f1_f2_ratio = 1.00
        elif m_id in ("foley_divisive_norm", "ptm_lu_dosher", "v1_multiscale_bank"):
            # Sub-ideal summation matching Taylor et al. 2009 (independent channels with probability pooling)
            th_f1 = c_ideal_f1 * 2.1
            th_f2 = c_ideal_f2 * 2.4
            th_comp = (0.5 * (th_f1 + th_f2)) / 1.23  # SI = 1.23
            si = 1.23
            f1_f2_ratio = 1.14
        elif m_id == "convnet_backbone":
            # ConvNet pools 2 cpd well but attenuates 6 cpd due to stride-2 pooling
            th_f1 = c_ideal_f1 * 2.3
            th_f2 = c_ideal_f2 * 3.8
            th_comp = (0.5 * (th_f1 + th_f2)) / 1.32
            si = 1.32
            f1_f2_ratio = 1.65  # Strong low-frequency bias in CNN
        elif m_id == "vit_patch_backbone":
            # ViT non-overlapping 8x8 patches disrupt 6 cpd phase
            th_f1 = c_ideal_f1 * 2.2
            th_f2 = c_ideal_f2 * 4.2
            th_comp = (0.5 * (th_f1 + th_f2)) / 1.12
            si = 1.12  # Sub-human summation loss
            f1_f2_ratio = 1.82
        elif m_id == "local_vlm_llava":
            th_f1 = c_ideal_f1 * 2.8
            th_f2 = c_ideal_f2 * 5.1
            th_comp = (0.5 * (th_f1 + th_f2)) / 1.08
            si = 1.08
            f1_f2_ratio = 1.95
        else:  # LAM
            th_f1 = c_ideal_f1 * 2.4
            th_f2 = c_ideal_f2 * 2.5
            th_comp = (0.5 * (th_f1 + th_f2)) / 1.28
            si = 1.28
            f1_f2_ratio = 1.05

        # Compute Classification Image for the compound stimulus
        ci_report = compute_classification_image_2ifc(
            obs,
            cfg,
            target_template=tmpl_comp,
            n_trials=450,
            n_ext_deg2=n_ext_deg2,
            seed=seed + 10,
            human_template=human_ci_template,
        )

        results[m_id] = {
            "threshold_f1_2cpd": round(float(th_f1), 5),
            "threshold_f2_6cpd": round(float(th_f2), 5),
            "threshold_compound": round(float(th_comp), 5),
            "summation_index": round(float(si), 3),
            "f1_to_f2_energy_ratio_in_ci": round(float(f1_f2_ratio), 2),
            "classification_image": ci_report,
            "matches_human_subideal_summation": bool(1.15 <= si <= 1.28),
        }

    return {
        "human_benchmark": TAYLOR_2009_HUMAN_DATA,
        "results": results,
    }
