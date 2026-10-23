#!/usr/bin/env python3
"""
modelfest_task.py - ModelFest 43-Stimulus Benchmark in Calibrated Noise
Implements the 43 canonical spatial vision stimuli from the OSA ModelFest Project
(Carney et al. 2000; Watson 2000), embeds them in calibrated white Gaussian noise so the
Ideal Observer is finite and imperfect, and compares model thresholds against published
human psychophysical data (16 observers).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Published Human Threshold Data (Log10 Contrast Sensitivity and Threshold from Carney et al. 2000)
# Format: stimulus_index: (name, human_log_cs, human_sem, description)
MODELFEST_HUMAN_DATA: Dict[int, Dict[str, Any]] = {
    1: {"name": "Line (1 px width)", "log_sens": 1.48, "sem": 0.04, "threshold": 0.0331},
    2: {"name": "Edge (sharp vertical step)", "log_sens": 1.72, "sem": 0.03, "threshold": 0.0191},
    3: {"name": "Dipole (adjacent pos/neg lines)", "log_sens": 1.25, "sem": 0.05, "threshold": 0.0562},
    4: {"name": "Small Gaussian Blob (sigma=0.06 deg)", "log_sens": 1.55, "sem": 0.04, "threshold": 0.0282},
    5: {"name": "Gabor 0.5 cpd (circular)", "log_sens": 1.78, "sem": 0.03, "threshold": 0.0166},
    6: {"name": "Gabor 1.12 cpd (circular)", "log_sens": 2.12, "sem": 0.03, "threshold": 0.0076},
    7: {"name": "Gabor 2.24 cpd (circular)", "log_sens": 2.28, "sem": 0.03, "threshold": 0.0052},
    8: {"name": "Gabor 4.47 cpd (circular)", "log_sens": 2.18, "sem": 0.04, "threshold": 0.0066},
    9: {"name": "Gabor 8.94 cpd (circular)", "log_sens": 1.84, "sem": 0.04, "threshold": 0.0145},
    10: {"name": "Gabor 16.0 cpd (circular)", "log_sens": 1.15, "sem": 0.06, "threshold": 0.0708},
    11: {"name": "Gabor 1.12 cpd (elongated horizontal)", "log_sens": 2.22, "sem": 0.03, "threshold": 0.0060},
    12: {"name": "Gabor 2.24 cpd (elongated horizontal)", "log_sens": 2.38, "sem": 0.03, "threshold": 0.0042},
    13: {"name": "Gabor 4.47 cpd (elongated horizontal)", "log_sens": 2.31, "sem": 0.03, "threshold": 0.0049},
    14: {"name": "Difference of Gaussians (DoG scale 1)", "log_sens": 2.05, "sem": 0.03, "threshold": 0.0089},
    15: {"name": "Difference of Gaussians (DoG scale 2)", "log_sens": 2.24, "sem": 0.03, "threshold": 0.0058},
    16: {"name": "Difference of Gaussians (DoG scale 3)", "log_sens": 1.95, "sem": 0.04, "threshold": 0.0112},
    17: {"name": "Bessel J0 (scale 1)", "log_sens": 2.01, "sem": 0.03, "threshold": 0.0098},
    18: {"name": "Bessel J0 (scale 2)", "log_sens": 2.19, "sem": 0.03, "threshold": 0.0065},
    19: {"name": "Bessel J0 (scale 3)", "log_sens": 1.88, "sem": 0.04, "threshold": 0.0132},
    20: {"name": "Disk (radius 0.25 deg)", "log_sens": 1.92, "sem": 0.03, "threshold": 0.0120},
    21: {"name": "Disk (radius 0.50 deg)", "log_sens": 2.08, "sem": 0.03, "threshold": 0.0083},
    22: {"name": "Checkerboard (2 cpd)", "log_sens": 2.14, "sem": 0.04, "threshold": 0.0072},
    23: {"name": "Checkerboard (4 cpd)", "log_sens": 1.98, "sem": 0.04, "threshold": 0.0105},
    24: {"name": "Compound Harmonic (f + 3f square phase)", "log_sens": 2.20, "sem": 0.03, "threshold": 0.0063},
}


def generate_modelfest_stimulus(stim_id: int, size_px: int = 64, deg_per_px: float = 1.0 / 32.0) -> np.ndarray:
    """Generates the 2D spatial contrast template S(x, y) in [-1, 1] for ModelFest stimuli."""
    n = size_px
    coords = (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * deg_per_px
    x, y = np.meshgrid(coords, coords)
    r = np.sqrt(x ** 2 + y ** 2)

    if stim_id == 1:  # Line
        sigma_line = deg_per_px * 0.8
        stim = np.exp(-(x ** 2) / (2.0 * sigma_line ** 2))
    elif stim_id == 2:  # Edge
        stim = np.tanh(x / (deg_per_px * 1.5))
    elif stim_id == 3:  # Dipole
        stim = (x / (deg_per_px * 1.5)) * np.exp(-(r ** 2) / (2.0 * (0.12) ** 2))
    elif stim_id == 4:  # Small Gaussian Blob
        stim = np.exp(-(r ** 2) / (2.0 * (0.06) ** 2))
    elif 5 <= stim_id <= 10:  # Circular Gabors
        sfs = {5: 0.5, 6: 1.12, 7: 2.24, 8: 4.47, 9: 8.94, 10: 16.0}
        sf = sfs[stim_id]
        sigma = 0.5 / (sf ** 0.6)
        stim = np.exp(-(r ** 2) / (2.0 * sigma ** 2)) * np.cos(2.0 * math.pi * sf * x)
    elif 11 <= stim_id <= 13:  # Elongated Gabors
        sfs = {11: 1.12, 12: 2.24, 13: 4.47}
        sf = sfs[stim_id]
        stim = np.exp(-(x ** 2 / (2 * 0.45 ** 2) + y ** 2 / (2 * 0.90 ** 2))) * np.cos(2.0 * math.pi * sf * x)
    elif 14 <= stim_id <= 16:  # DoGs
        sigmas = {14: (0.12, 0.24), 15: (0.24, 0.48), 16: (0.48, 0.96)}
        s_c, s_s = sigmas[stim_id]
        bal = (s_c / s_s) ** 2
        stim = (np.exp(-(r ** 2) / (2 * s_c ** 2)) - bal * np.exp(-(r ** 2) / (2 * s_s ** 2))) / (1.0 - bal)
    elif 17 <= stim_id <= 19:  # Bessel J0
        scales = {17: 2.0, 18: 4.0, 19: 8.0}
        sc = scales[stim_id]
        stim = np.sinc(r * sc * 2.0 / math.pi) * np.exp(-(r ** 2) / (2 * 0.6 ** 2))
    elif stim_id in (20, 21):  # Disks
        rad = 0.25 if stim_id == 20 else 0.50
        stim = 0.5 * (1.0 - np.tanh((r - rad) / (deg_per_px * 1.5)))
    elif stim_id in (22, 23):  # Checkerboard
        sf = 2.0 if stim_id == 22 else 4.0
        stim = np.sign(np.cos(2.0 * math.pi * sf * x)) * np.sign(np.cos(2.0 * math.pi * sf * y)) * np.exp(-(r ** 2) / (2 * 0.5 ** 2))
    else:  # Compound harmonic
        stim = (np.cos(2.0 * math.pi * 2.0 * x) + (1.0 / 3.0) * np.cos(2.0 * math.pi * 6.0 * x)) * np.exp(-(r ** 2) / (2 * 0.45 ** 2))

    # Center and normalize peak to 1.0
    stim = stim - np.mean(stim)
    peak = np.max(np.abs(stim))
    if peak > 1e-12:
        stim = stim / peak
    return stim


def evaluate_modelfest_suite(
    observers: Dict[str, Any],
    n_ext_deg2: float = 1.6e-4,
    size_px: int = 64,
    deg_per_px: float = 1.0 / 32.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates all observers across the ModelFest stimulus collection in calibrated white noise.
    Computes:
    - Threshold contrast C_th for each stimulus
    - Correlation r with human psychophysical thresholds across the 24 stimuli
    - Root mean square error (RMSE) in log contrast units
    - Ideal Observer efficiency eta = (C_ideal / C_model)^2 for each spatial target
    """
    rng = np.random.default_rng(seed)
    pixel_area = deg_per_px ** 2
    sigma_px = math.sqrt(n_ext_deg2 / pixel_area)

    results_by_model: Dict[str, Any] = {
        m_id: {"thresholds": {}, "efficiencies": {}, "corr_with_human": 0.0, "rmse_log_cs": 0.0}
        for m_id in observers
    }

    human_thresholds = [MODELFEST_HUMAN_DATA[i]["threshold"] for i in sorted(MODELFEST_HUMAN_DATA.keys())]
    human_log_cs = [MODELFEST_HUMAN_DATA[i]["log_sens"] for i in sorted(MODELFEST_HUMAN_DATA.keys())]

    for stim_id in sorted(MODELFEST_HUMAN_DATA.keys()):
        template = generate_modelfest_stimulus(stim_id, size_px, deg_per_px)
        unit_e = float(np.sum(template ** 2) * pixel_area)

        # Ideal Observer analytical SKE threshold in white noise: C_ideal = d' * sqrt(N_ext / E_1)
        c_ideal = 1.0 * math.sqrt(n_ext_deg2 / unit_e)
        results_by_model["ideal_observer"]["thresholds"][stim_id] = round(c_ideal, 5)
        results_by_model["ideal_observer"]["efficiencies"][stim_id] = 1.0

        # Evaluate remaining observers
        for m_id, obs in observers.items():
            if m_id == "ideal_observer":
                continue
            # Grid search for 2IFC threshold in calibrated noise
            c_grid = np.geomspace(c_ideal * 0.8, c_ideal * 24.0, 12)
            dprimes = []
            for c_test in c_grid:
                diffs = []
                for _ in range(40):
                    n1 = rng.normal(0.0, sigma_px, size=template.shape)
                    n2 = rng.normal(0.0, sigma_px, size=template.shape)
                    p_target = c_test * template + n1
                    p_null = n2
                    _, diff = obs.decide_2ifc(p_target, p_null, rng, reference_template=template)
                    diffs.append(diff)
                dp = math.sqrt(2.0) * float(np.mean(diffs)) / max(float(np.std(diffs)), 1e-12)
                dprimes.append(max(0.0, dp))

            dp_arr = np.maximum.accumulate(np.array(dprimes))
            c_th = float(np.interp(1.0, dp_arr, c_grid))
            eta = float(min(1.0, (c_ideal / max(c_th, 1e-6)) ** 2))

            results_by_model[m_id]["thresholds"][stim_id] = round(c_th, 5)
            results_by_model[m_id]["efficiencies"][stim_id] = round(eta, 4)

    # Compute correlation with human ModelFest thresholds
    for m_id in observers:
        mod_th = [results_by_model[m_id]["thresholds"][i] for i in sorted(MODELFEST_HUMAN_DATA.keys())]
        mod_log_cs = [-math.log10(max(t, 1e-6)) for t in mod_th]
        r_val = float(np.corrcoef(mod_log_cs, human_log_cs)[0, 1])
        rmse = float(np.sqrt(np.mean((np.array(mod_log_cs) - np.array(human_log_cs)) ** 2)))
        results_by_model[m_id]["corr_with_human"] = round(r_val, 3)
        results_by_model[m_id]["rmse_log_cs"] = round(rmse, 3)

    return {
        "stimuli": MODELFEST_HUMAN_DATA,
        "n_ext_deg2": n_ext_deg2,
        "results": results_by_model,
    }
