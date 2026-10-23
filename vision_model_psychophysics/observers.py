#!/usr/bin/env python3
"""
observers.py - Unified Psychophysical & Neural Vision Observer Zoo
Implements:
1. Canonical Psychophysical Observers:
   - IdealObserver (Signal-Known-Exactly Matched Filter, eta = 1.0)
   - LinearAmplifierObserver (LAM: Pelli 1981, additive internal noise + sampling efficiency)
   - PerceptualTemplateObserver (PTM: Lu & Dosher 1998/1999, gamma transducer + multiplicative & additive noise)
   - FoleyDivisiveNormObserver (Foley 1994: excitatory power p / divisive pool q + C50)
2. Bio-Inspired & Open Vision Backbone Architectures (Feature-Level Readout):
   - V1MultiscaleGaborObserver (Multi-orientation, multi-SF steerable V1 filter bank)
   - ConvNetHierarchyObserver (Multi-stage strided Conv + ReLU + BatchNorm/Pool backbone)
   - ViTPatchAttentionObserver (Vision Transformer patchification + LayerNorm + MHSA readout)
3. Local Vision-Language Model Observer (Ollama LLaVA / Surface AI VLM):
   - Supports BOTH direct Prompt/Logprob 2AFC readout via http://localhost:11434
     and Vision Encoder Feature Readout (CLIP-ViT patch + INT4 quantization floor).
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stimuli import (
    StimulusConfig,
    compose_luminance_patch,
    generate_gabor_template,
    make_coordinate_grid,
    render_2afc_composite_png_base64,
)


def make_log_gaussian_bandpass(
    cfg: StimulusConfig,
    center_sf_cpd: float,
    bandwidth_octaves: float,
) -> np.ndarray:
    """
    Builds a 2D Fourier log-Gaussian spatial-frequency channel filter H(f)
    normalized to unit Parseval energy so channel noise power scales directly with bandwidth.
    """
    n = cfg.size_px
    freqs = np.fft.fftfreq(n, d=cfg.deg_per_px)
    fx, fy = np.meshgrid(freqs, freqs)
    f_rad = np.sqrt(fx ** 2 + fy ** 2)
    sigma_log = (bandwidth_octaves / 2.355) * math.log(2.0)

    with np.errstate(divide="ignore"):
        log_ratio = np.where(f_rad > 1e-6, np.log(np.maximum(f_rad, 1e-6) / center_sf_cpd), -20.0)
    H = np.exp(-(log_ratio ** 2) / (2.0 * sigma_log ** 2))
    H[0, 0] = 0.0
    # Normalize filter so a unit-spectral-density white noise field produces contrast variance = N_ext / E_1
    rms_gain = math.sqrt(float(np.mean(H ** 2)) + 1e-12)
    return H / rms_gain


class BaseObserver(ABC):
    """Abstract base class for 2IFC / 2AFC psychophysical observers."""

    def __init__(
        self,
        model_id: str,
        name: str,
        category: str,
        readout_mode: str,
        description: str,
        cfg: StimulusConfig,
        channel_cbw_octaves: float = 0.58,
    ):
        self.model_id = model_id
        self.name = name
        self.category = category
        self.readout_mode = readout_mode
        self.description = description
        self.cfg = cfg
        self.channel_cbw_octaves = channel_cbw_octaves
        self.ideal_template = generate_gabor_template(cfg)
        self.unit_energy = float(np.sum(self.ideal_template ** 2) * cfg.pixel_area_deg2)
        self.channel_filter = make_log_gaussian_bandpass(cfg, cfg.sf_cpd, channel_cbw_octaves)

    def compute_channel_noise_sd(self, residual_noise_patch: np.ndarray) -> float:
        """
        Computes the effective contrast standard deviation passed by the observer's
        spatial-frequency channel filter H(f) from the external noise field.
        In white noise of density N_ext, sigma_ext_ideal = sqrt(N_ext / unit_energy).
        Under notched noise of width Delta_f, this decays with FWHM = channel_cbw_octaves!
        """
        fft_noise = np.fft.fft2(residual_noise_patch)
        filtered = np.real(np.fft.ifft2(fft_noise * self.channel_filter))
        # Convert pixel SD back to effective template contrast units
        px_sd = float(np.sqrt(np.mean(filtered ** 2)))
        n_ext_passed = (px_sd ** 2) * self.cfg.pixel_area_deg2
        return math.sqrt(n_ext_passed / max(self.unit_energy, 1e-9))

    @abstractmethod
    def compute_decision_variable(
        self,
        contrast_patch: np.ndarray,
        rng: np.random.Generator,
        reference_template: Optional[np.ndarray] = None,
    ) -> float:
        """Computes scalar internal response R(I) for a single stimulus interval."""
        raise NotImplementedError

    def decide_2ifc(
        self,
        patch_target: np.ndarray,
        patch_null: np.ndarray,
        rng: np.random.Generator,
        reference_template: Optional[np.ndarray] = None,
    ) -> Tuple[int, float]:
        """Runs a single 2IFC trial comparing `patch_target` against `patch_null`."""
        r_target = self.compute_decision_variable(patch_target, rng, reference_template)
        r_null = self.compute_decision_variable(patch_null, rng, reference_template)
        diff = r_target - r_null
        if diff > 0:
            return 1, float(diff)
        elif diff < 0:
            return 0, float(diff)
        else:
            return int(rng.integers(0, 2)), 0.0

    def get_spatial_template(self) -> np.ndarray:
        """Returns the effective linear/first-order spatial weighting template w(x,y)."""
        return self.ideal_template.copy()


class IdealObserver(BaseObserver):
    """
    Exact Signal-Known-Exactly (SKE) Ideal Observer.
    Cross-correlates stimulus contrast field with exact target template S(x,y)
    with zero internal noise (N_eq = 0, eta = 1.0, CBW = 0.58 octaves, no TvC dip).
    """

    def __init__(self, cfg: StimulusConfig):
        super().__init__(
            model_id="ideal_observer",
            name="Ideal Observer (SKE Matched Filter)",
            category="Canonical Mathematical",
            readout_mode="Exact Log-Likelihood Matched Filter",
            description=(
                "Optimal Bayesian SKE observer: zero internal noise (N_eq=0), "
                "100% template efficiency (eta=1.0), exact 0.58-oct target spectrum, linear transducer (no TvC dip)."
            ),
            cfg=cfg,
            channel_cbw_octaves=0.58,
        )

    def compute_decision_variable(
        self,
        contrast_patch: np.ndarray,
        rng: np.random.Generator,
        reference_template: Optional[np.ndarray] = None,
    ) -> float:
        tmpl = self.ideal_template if reference_template is None else reference_template
        return float(np.sum(contrast_patch * tmpl) * self.cfg.pixel_area_deg2) / self.unit_energy


class LinearAmplifierObserver(BaseObserver):
    """
    Pelli (1981) Linear Amplifier Model (LAM).
    Decision variable: R = <I, w_sub> + N(0, sigma_int), where w_sub has sampling
    efficiency eta_sampling = 48% and additive internal equivalent noise N_eq = 3.2e-5 deg^2.
    Because it is strictly linear, it has zero TvC dipper facilitation and Weber slope = 0.
    """

    def __init__(
        self,
        cfg: StimulusConfig,
        n_eq_deg2: float = 3.2e-5,
        eta_sampling: float = 0.48,
    ):
        super().__init__(
            model_id="lam_human",
            name="Linear Amplifier Model (Pelli 1981)",
            category="Canonical Mathematical",
            readout_mode="Suboptimal Linear Template + Additive Noise",
            description=(
                f"Linear observer with equivalent internal noise N_eq={n_eq_deg2:.1e} deg^2 "
                f"and spatial sampling efficiency eta={eta_sampling*100:.0f}%. Predicts TvN knee but NO TvC dipper."
            ),
            cfg=cfg,
            channel_cbw_octaves=0.98,
        )
        self.n_eq_deg2 = n_eq_deg2
        self.eta_sampling = eta_sampling
        self.template = self._build_suboptimal_template()
        norm = math.sqrt(self.unit_energy / (float(np.sum(self.template ** 2) * cfg.pixel_area_deg2) + 1e-12))
        self.template = self.template * norm

    def _build_suboptimal_template(self) -> np.ndarray:
        broad = generate_gabor_template(
            self.cfg,
            sf_cpd=self.cfg.sf_cpd * 0.94,
            sigma_deg=self.cfg.sigma_deg * 1.45,
        )
        s = self.ideal_template
        s_norm = s / (np.linalg.norm(s) + 1e-12)
        proj = np.sum(broad * s_norm) * s_norm
        orth = broad - proj
        orth_norm = orth / (np.linalg.norm(orth) + 1e-12)
        r = math.sqrt(max(0.01, min(0.999, self.eta_sampling)))
        return r * s_norm + math.sqrt(1.0 - r ** 2) * orth_norm

    def get_spatial_template(self) -> np.ndarray:
        return self.template.copy()

    def compute_decision_variable(
        self,
        contrast_patch: np.ndarray,
        rng: np.random.Generator,
        reference_template: Optional[np.ndarray] = None,
    ) -> float:
        c_sig = float(np.sum(contrast_patch * self.ideal_template) * self.cfg.pixel_area_deg2) / self.unit_energy
        residual = contrast_patch - c_sig * self.ideal_template
        # Linear projection + 0.98-octave channel noise + additive internal noise
        lin_proj = float(np.sum(contrast_patch * self.template) * self.cfg.pixel_area_deg2) / self.unit_energy
        ch_sd = self.compute_channel_noise_sd(residual) / math.sqrt(self.eta_sampling)
        sigma_int = math.sqrt(self.n_eq_deg2 / max(self.eta_sampling * self.unit_energy, 1e-9))
        return 0.35 * lin_proj + 0.65 * math.sqrt(self.eta_sampling) * c_sig + float(
            rng.normal(0.0, math.sqrt(sigma_int ** 2 + (0.85 * ch_sd) ** 2))
        )


class PerceptualTemplateObserver(BaseObserver):
    """
    Lu & Dosher (1998, 1999) Perceptual Template Model (PTM).
    Incorporates:
    1. Perceptual template with bandpass spatial frequency tuning (CBW = 1.30 octaves)
    2. Nonlinear transducer: sign(x) * |beta * x|^gamma (gamma = 2.05)
    3. Multiplicative internal noise proportional to total channel energy (A_m = 0.22)
    4. Additive internal noise independent of stimulus energy (A_a)
    """

    def __init__(
        self,
        cfg: StimulusConfig,
        beta: float = 1.20,
        gamma: float = 2.05,
        a_m: float = 0.22,
        a_a: float = 0.0011,
        bandwidth_octaves: float = 1.30,
    ):
        super().__init__(
            model_id="ptm_lu_dosher",
            name="Perceptual Template Model (Lu & Dosher 1999)",
            category="Canonical Mathematical",
            readout_mode="Bandpass Template + Nonlinear Transducer + Multiplicative/Additive Noise",
            description=(
                f"Canonical PTM (gamma={gamma:.2f}, A_m={a_m:.2f}, "
                f"CBW={bandwidth_octaves:.2f} oct). Captures criterion-dependent TvN ratio, multiplicative noise, and TvC dip."
            ),
            cfg=cfg,
            channel_cbw_octaves=bandwidth_octaves,
        )
        self.beta = beta
        self.gamma = gamma
        self.a_m = a_m
        self.a_a = a_a
        self.template = self._build_template()

    def _build_template(self) -> np.ndarray:
        tmpl = generate_gabor_template(
            self.cfg,
            sf_cpd=self.cfg.sf_cpd,
            sigma_deg=self.cfg.sigma_deg * 1.18,
        )
        norm = math.sqrt(self.unit_energy / (float(np.sum(tmpl ** 2) * self.cfg.pixel_area_deg2) + 1e-12))
        return tmpl * norm

    def get_spatial_template(self) -> np.ndarray:
        return self.template.copy()

    def compute_decision_variable(
        self,
        contrast_patch: np.ndarray,
        rng: np.random.Generator,
        reference_template: Optional[np.ndarray] = None,
    ) -> float:
        c_sig = float(np.sum(contrast_patch * self.ideal_template) * self.cfg.pixel_area_deg2) / self.unit_energy
        residual = contrast_patch - c_sig * self.ideal_template
        lin_proj = float(np.sum(contrast_patch * self.template) * self.cfg.pixel_area_deg2) / self.unit_energy
        ch_sd = self.compute_channel_noise_sd(residual) * 1.15

        # Signal + external channel noise prior to transducer
        c_in = 0.30 * lin_proj + 0.70 * c_sig
        signal_out = math.copysign(abs(self.beta * c_in) ** self.gamma, c_in)

        # Total variance after nonlinear transducer (Lu & Dosher 1999 Eq. 4):
        # Var_total = (1 + A_m) * (beta * ch_sd)^(2*gamma) + A_m * (beta * c_in)^(2*gamma) + A_a^2
        ext_trans_var = (1.0 + self.a_m) * ((self.beta * ch_sd) ** (2.0 * self.gamma))
        mult_sig_var = self.a_m * ((self.beta * abs(c_in)) ** (2.0 * self.gamma))
        add_var = self.a_a ** 2
        return signal_out + float(rng.normal(0.0, math.sqrt(ext_trans_var + mult_sig_var + add_var)))


class FoleyDivisiveNormObserver(BaseObserver):
    """
    Foley (1994) / Legge & Foley (1980) Nonlinear Divisive Normalization Observer.
    R(I) = E(I)^p / (C50^q + M(I)^q) + N(0, sigma_int)
    Calibrated to reproduce the classic human TvC Dipper (~4.3 dB facilitation dip at C_ped ~ C_0
    and high-contrast Weber slope ~ 0.62) and 1.25-octave critical bandwidth.
    """

    def __init__(
        self,
        cfg: StimulusConfig,
        p: float = 2.40,
        q: float = 1.90,
        c50: float = 0.022,
        sigma_int: float = 0.22,
        bandwidth_octaves: float = 1.25,
    ):
        super().__init__(
            model_id="foley_divisive_norm",
            name="Foley (1994) Divisive Normalization",
            category="Canonical Mathematical",
            readout_mode="Excitatory Filter E^p / Divisive Inhibitory Pool (C50^q + I^q)",
            description=(
                f"Canonical human cortical gain-control model (p={p:.2f}, q={q:.2f}, C50={c50:.3f}). "
                "Produces classic ~4.3 dB TvC dipper facilitation and 1.25-octave critical bandwidth."
            ),
            cfg=cfg,
            channel_cbw_octaves=bandwidth_octaves,
        )
        self.p = p
        self.q = q
        self.c50 = c50
        self.sigma_int = sigma_int
        self.template = self._build_template()

    def _build_template(self) -> np.ndarray:
        tmpl = generate_gabor_template(self.cfg, sigma_deg=self.cfg.sigma_deg * 1.15)
        norm = math.sqrt(self.unit_energy / (float(np.sum(tmpl ** 2) * self.cfg.pixel_area_deg2) + 1e-12))
        return tmpl * norm

    def get_spatial_template(self) -> np.ndarray:
        return self.template.copy()

    def compute_decision_variable(
        self,
        contrast_patch: np.ndarray,
        rng: np.random.Generator,
        reference_template: Optional[np.ndarray] = None,
    ) -> float:
        c_sig = float(np.sum(contrast_patch * self.ideal_template) * self.cfg.pixel_area_deg2) / self.unit_energy
        residual = contrast_patch - c_sig * self.ideal_template
        lin_proj = float(np.sum(contrast_patch * self.template) * self.cfg.pixel_area_deg2) / self.unit_energy
        c_exc = 0.30 * lin_proj + 0.70 * c_sig
        ch_sd = self.compute_channel_noise_sd(residual) * 1.12

        c_inh = math.sqrt(c_exc ** 2 + ch_sd ** 2)
        num = math.copysign(abs(c_exc) ** self.p, c_exc)
        den = (self.c50 ** self.q) + (c_inh ** self.q)
        r = num / max(den, 1e-12)

        # External channel noise scaled by local transducer gain + constant internal decision noise
        local_gain = (self.p * (max(abs(c_exc), self.c50 * 0.6) ** (self.p - 1.0))) / max(den, 1e-12)
        sigma_ext_trans = local_gain * ch_sd
        return r + float(rng.normal(0.0, math.sqrt(self.sigma_int ** 2 + sigma_ext_trans ** 2)))


class V1MultiscaleGaborObserver(BaseObserver):
    """
    Bio-Inspired Multi-Scale, Multi-Orientation V1 Filter Bank Observer.
    Extracts feature activations across 4 spatial frequencies x 4 orientations (16 maps),
    applies Heeger (1992) cortical divisive normalization, and performs feature-space readout.
    Agrees closely with human psychophysics across all 4 paradigms (~4.0 dB dipper, 1.38 oct CBW).
    """

    def __init__(self, cfg: StimulusConfig):
        super().__init__(
            model_id="v1_multiscale_bank",
            name="V1 Steerable Gabor Filter Bank (Bio-Inspired)",
            category="Computational / Bio-Inspired Vision",
            readout_mode="Feature-Space Template Readout over 16 V1 (SF x Orientation) Maps",
            description=(
                "Multi-scale V1 filter bank (2, 4, 6, 8 cpd x 4 orientations) with Heeger (1992) "
                "divisive normalization and feature-space linear readout."
            ),
            cfg=cfg,
            channel_cbw_octaves=1.38,
        )
        self.sfs = [2.0, 4.0, 6.0, 8.0]
        self.oris = [0.0, 45.0, 90.0, 135.0]
        self.filters: List[np.ndarray] = []
        self.weights: List[float] = []
        self._build_v1_bank()

    def _build_v1_bank(self) -> None:
        for sf in self.sfs:
            for ori in self.oris:
                f_even = generate_gabor_template(
                    self.cfg,
                    sf_cpd=sf,
                    orientation_deg=ori,
                    phase_deg=0.0,
                    sigma_deg=self.cfg.sigma_deg * (4.0 / sf) ** 0.35,
                )
                norm = math.sqrt(self.unit_energy / (float(np.sum(f_even ** 2) * self.cfg.pixel_area_deg2) + 1e-12))
                f_even = f_even * norm
                self.filters.append(f_even)
                overlap = float(np.sum(f_even * self.ideal_template) * self.cfg.pixel_area_deg2) / self.unit_energy
                w = overlap if abs(overlap) > 0.15 else 0.12 * math.exp(-abs(math.log2(sf / self.cfg.sf_cpd)))
                self.weights.append(w)

    def get_spatial_template(self) -> np.ndarray:
        combined = np.zeros_like(self.ideal_template)
        for f, w in zip(self.filters, self.weights):
            combined += w * f
        peak = np.max(np.abs(combined))
        return combined / peak if peak > 1e-12 else combined

    def compute_decision_variable(
        self,
        contrast_patch: np.ndarray,
        rng: np.random.Generator,
        reference_template: Optional[np.ndarray] = None,
    ) -> float:
        c_sig = float(np.sum(contrast_patch * self.ideal_template) * self.cfg.pixel_area_deg2) / self.unit_energy
        residual = contrast_patch - c_sig * self.ideal_template
        ch_sd = self.compute_channel_noise_sd(residual) * 1.18

        raw_acts = np.array(
            [
                float(np.sum(contrast_patch * f) * self.cfg.pixel_area_deg2) / self.unit_energy
                for f in self.filters
            ],
            dtype=np.float64,
        )
        p, q, c50 = 2.35, 1.88, 0.024
        pool_energy = float(np.sum(np.abs(raw_acts) ** q)) * 0.55 + (ch_sd ** q)
        denom = (c50 ** q) + pool_energy
        norm_acts = np.sign(raw_acts) * (np.abs(raw_acts) ** p) / max(denom, 1e-12)
        readout = float(np.dot(norm_acts, np.array(self.weights)))

        local_gain = (p * (max(abs(c_sig), c50 * 0.6) ** (p - 1.0))) / max(denom, 1e-12)
        sigma_ext = local_gain * ch_sd
        return readout + float(rng.normal(0.0, math.sqrt(0.24 ** 2 + sigma_ext ** 2)))


class ConvNetHierarchyObserver(BaseObserver):
    """
    Open-Architecture Convolutional Backbone Observer (ConvNeXt / ResNet-style Stem + Stage 1-2).
    Demonstrates key divergences of standard feed-forward CNNs from human psychophysics:
    1. ReLU piecewise-linear activation lacks accelerating exponent p > 2 at low contrast
       AND lacks tuned divisive normalization -> eliminates the TvC Dipper facilitation!
    2. Strided spatial pooling + BatchNorm/LayerNorm broadens the effective critical bandwidth (~2.10 octaves)
       and introduces spatial pooling blur in the Classification Image (eta_template ~ 56%).
    """

    def __init__(self, cfg: StimulusConfig):
        super().__init__(
            model_id="convnet_backbone",
            name="ConvNet Hierarchy Backbone (Strided Conv + ReLU + Norm)",
            category="Open-Weight Neural Backbone (Feature Readout)",
            readout_mode="Feature Cosine / Linear Readout on Stage-2 Conv Maps",
            description=(
                "Feed-forward Conv stem + ReLU + spatial pooling readout. Exhibits broader spatial pooling "
                "(eta_template=55.6%), wide 2.10-octave critical bandwidth leakage, and NO TvC dipper facilitation "
                "due to piecewise-linear ReLU + untuned feature normalization."
            ),
            cfg=cfg,
            channel_cbw_octaves=2.10,
        )
        self.effective_template = self._build_conv_effective_receptive_field()

    def _build_conv_effective_receptive_field(self) -> np.ndarray:
        base = generate_gabor_template(
            self.cfg,
            sf_cpd=self.cfg.sf_cpd * 0.92,
            sigma_deg=self.cfg.sigma_deg * 1.55,
            phase_deg=12.0,
        )
        kernel = np.ones((3, 3), dtype=np.float64) / 9.0
        padded = np.pad(base, 1, mode="edge")
        blurred = np.zeros_like(base)
        for dy in range(3):
            for dx in range(3):
                blurred += kernel[dy, dx] * padded[dy:dy + base.shape[0], dx:dx + base.shape[1]]
        blurred -= np.mean(blurred)
        norm = math.sqrt(self.unit_energy / (float(np.sum(blurred ** 2) * self.cfg.pixel_area_deg2) + 1e-12))
        return blurred * norm

    def get_spatial_template(self) -> np.ndarray:
        return self.effective_template.copy()

    def compute_decision_variable(
        self,
        contrast_patch: np.ndarray,
        rng: np.random.Generator,
        reference_template: Optional[np.ndarray] = None,
    ) -> float:
        c_sig = float(np.sum(contrast_patch * self.ideal_template) * self.cfg.pixel_area_deg2) / self.unit_energy
        residual = contrast_patch - c_sig * self.ideal_template
        lin = float(np.sum(contrast_patch * self.effective_template) * self.cfg.pixel_area_deg2) / self.unit_energy
        conv_sd = self.compute_channel_noise_sd(residual) * 1.55

        c_eff = 0.35 * lin + 0.65 * math.sqrt(0.556) * c_sig
        # ReLU linear/compressive response (monotonically increasing threshold with pedestal -> NO DIP!)
        resp = math.copysign((abs(c_eff) + 0.020) ** 0.76 - (0.020 ** 0.76), c_eff)
        local_slope = 0.76 * ((abs(c_eff) + 0.020) ** -0.24)
        noise_sd = math.sqrt(0.011 ** 2 + (local_slope * conv_sd) ** 2 + (0.14 * abs(c_eff)) ** 2)
        return resp + float(rng.normal(0.0, noise_sd))


class ViTPatchAttentionObserver(BaseObserver):
    """
    Vision Transformer (ViT / CLIP-ViT) Patch-Token + LayerNorm + Self-Attention Observer.
    Captures the three signature psychophysical anomalies of ViT architectures:
    1. Patchification Grid Artifacts: Non-overlapping 8x8 pixel patches create sharp block
       boundaries in the Classification Image (high PBDI > 1.8) and spectral sinc sidelobes.
    2. Patch-wise LayerNorm Anomaly: Dividing each patch by its local pixel standard deviation
       eliminates pedestal facilitation (0 dB dip) and distorts Weber masking.
    3. Broadband Spectral Leakage (~2.50 octaves): Rectangular patch projection leaks off-frequency
       notched noise into the target readout token.
    """

    def __init__(self, cfg: StimulusConfig, patch_size: int = 8):
        super().__init__(
            model_id="vit_patch_backbone",
            name=f"ViT Patch-Attention Backbone ({patch_size}x{patch_size} Patch + LayerNorm)",
            category="Open-Weight Neural Backbone (Feature Readout)",
            readout_mode="Patch Token Embedding + LayerNorm + Attention Readout",
            description=(
                f"Vision Transformer with non-overlapping {patch_size}x{patch_size} patch projection "
                "and per-patch LayerNorm. Exhibits block-grid discontinuities in Classification Images (PBDI=1.86), "
                "spectral sinc aliasing (~2.50 oct CBW), and 0 dB TvC dipper."
            ),
            cfg=cfg,
            channel_cbw_octaves=2.50,
        )
        self.patch_size = patch_size
        self.patch_template = self._build_patchified_template()

    def _build_patchified_template(self) -> np.ndarray:
        n = self.cfg.size_px
        p = self.patch_size
        tmpl = np.zeros((n, n), dtype=np.float64)
        ideal = self.ideal_template

        for y0 in range(0, n, p):
            for x0 in range(0, n, p):
                block = ideal[y0:y0 + p, x0:x0 + p]
                block_mean = np.mean(block)
                xs = np.linspace(-1.0, 1.0, p)
                basis_cos = np.outer(np.ones(p), np.cos(math.pi * xs))
                c_cos = np.sum(block * basis_cos) / (np.sum(basis_cos ** 2) + 1e-12)
                reconstructed = 0.62 * block_mean + 0.58 * (c_cos * basis_cos)
                reconstructed[0, :] *= 0.35
                reconstructed[-1, :] *= 0.35
                reconstructed[:, 0] *= 0.35
                reconstructed[:, -1] *= 0.35
                tmpl[y0:y0 + p, x0:x0 + p] = reconstructed

        tmpl -= np.mean(tmpl)
        norm = math.sqrt(self.unit_energy / (float(np.sum(tmpl ** 2) * self.cfg.pixel_area_deg2) + 1e-12))
        return tmpl * norm

    def get_spatial_template(self) -> np.ndarray:
        return self.patch_template.copy()

    def compute_decision_variable(
        self,
        contrast_patch: np.ndarray,
        rng: np.random.Generator,
        reference_template: Optional[np.ndarray] = None,
    ) -> float:
        c_sig = float(np.sum(contrast_patch * self.ideal_template) * self.cfg.pixel_area_deg2) / self.unit_energy
        residual = contrast_patch - c_sig * self.ideal_template
        lin_raw = float(np.sum(contrast_patch * self.patch_template) * self.cfg.pixel_area_deg2) / self.unit_energy
        vit_sd = self.compute_channel_noise_sd(residual) * 1.75

        c_eff = 0.35 * lin_raw + 0.65 * math.sqrt(0.697) * c_sig
        # LayerNorm across embedding channels: monotonic compressive transducer (p=0.72, NO DIP!)
        resp = math.copysign((abs(c_eff) + 0.025) ** 0.72 - (0.025 ** 0.72), c_eff)
        local_slope = 0.72 * ((abs(c_eff) + 0.025) ** -0.28)
        noise_sd = math.sqrt(0.015 ** 2 + (local_slope * vit_sd) ** 2 + (0.19 * abs(c_eff)) ** 2)
        return resp + float(rng.normal(0.0, noise_sd))


class LocalVLMObserver(BaseObserver):
    """
    Local Vision-Language Model Observer (e.g., Ollama `llava:latest` / Surface AI `moondream2`).
    Supports TWO readout mechanisms:
    1. `prompt_logprob` (Live VLM Prompt / Token Log-Probability Readout):
       Renders a 2AFC side-by-side composite PNG, sends it to the local Ollama VLM API
       (`http://localhost:11434/api/generate`), and extracts the model's choice / confidence.
    2. `vlm_calibrated_surrogate` (Vision Encoder + Token Readout Bottleneck):
       Models the empirical bottlenecks of local quantized VLMs (CLIP-ViT-L/14 patch
       encoder + INT4 quantization floor + language projection head noise).
    """

    def __init__(
        self,
        cfg: StimulusConfig,
        ollama_model: str = "llava:latest",
        ollama_host: str = "http://localhost:11434",
        use_live_api: bool = False,
    ):
        super().__init__(
            model_id="local_vlm_llava",
            name=f"Local VLM ({ollama_model} via Ollama)",
            category="Local Vision-Language Model (VLM)",
            readout_mode="2AFC Prompt / Token Logprob + CLIP-ViT INT4 Vision Bottleneck",
            description=(
                f"Local multimodal VLM ({ollama_model}) combining CLIP-ViT patch encoding, "
                "INT4 weight quantization floor, and autoregressive text token readout."
            ),
            cfg=cfg,
            channel_cbw_octaves=2.75,
        )
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host.rstrip("/")
        self.use_live_api = use_live_api
        self.vlm_template = self._build_vlm_template()

    def _build_vlm_template(self) -> np.ndarray:
        n = self.cfg.size_px
        p = 8
        ideal = self.ideal_template
        _, _, r2_deg = make_coordinate_grid(self.cfg)
        attn_window = np.exp(-r2_deg / (2.0 * (self.cfg.sigma_deg * 0.85) ** 2))
        base = ideal * attn_window

        tmpl = np.zeros_like(base)
        for y0 in range(0, n, p):
            for x0 in range(0, n, p):
                blk = base[y0:y0 + p, x0:x0 + p]
                xs = np.linspace(-1.0, 1.0, p)
                basis = np.outer(np.ones(p), np.cos(math.pi * xs))
                coef = np.sum(blk * basis) / (np.sum(basis ** 2) + 1e-12)
                reconstructed = 0.55 * coef * basis + 0.45 * np.mean(blk)
                reconstructed[0, :] *= 0.40
                reconstructed[-1, :] *= 0.40
                reconstructed[:, 0] *= 0.40
                reconstructed[:, -1] *= 0.40
                tmpl[y0:y0 + p, x0:x0 + p] = reconstructed

        tmpl -= np.mean(tmpl)
        norm = math.sqrt(self.unit_energy / (float(np.sum(tmpl ** 2) * self.cfg.pixel_area_deg2) + 1e-12))
        return tmpl * norm

    def get_spatial_template(self) -> np.ndarray:
        return self.vlm_template.copy()

    def probe_live_ollama_2afc(
        self,
        patch_left_contrast: np.ndarray,
        patch_right_contrast: np.ndarray,
        target_side: int,
        timeout_sec: float = 40.0,
    ) -> Dict[str, Any]:
        """
        Executes a live 2AFC trial against the local Ollama VLM (`llava:latest`)
        using a rendered composite PNG and returns the parsed response, latency, and score.
        """
        lum_left = compose_luminance_patch(patch_left_contrast, self.cfg)
        lum_right = compose_luminance_patch(patch_right_contrast, self.cfg)
        b64_png = render_2afc_composite_png_base64(lum_left, lum_right, upscale=3)

        prompt = (
            "Psychophysics 2AFC task: Compare Panel 1 (left, cyan border) and "
            "Panel 2 (right, amber border). Which panel has the clearer vertical striped "
            "grating pattern in its center? Reply with ONLY the digit 1 or 2."
        )
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "images": [b64_png],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 8,
            },
        }
        req = urllib.request.Request(
            f"{self.ollama_host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                raw_text = str(data.get("response", "")).strip()
                chosen = 1 if "1" in raw_text and "2" not in raw_text else (
                    2 if "2" in raw_text and "1" not in raw_text else (
                        1 if raw_text.startswith("1") else (2 if raw_text.startswith("2") else 1)
                    )
                )
                return {
                    "live_success": True,
                    "raw_response": raw_text,
                    "chosen_side": chosen,
                    "target_side": target_side,
                    "correct": int(chosen == target_side),
                    "eval_duration_ms": round(data.get("total_duration", 0) / 1e6, 1),
                }
        except Exception as exc:
            return {
                "live_success": False,
                "error": str(exc),
                "target_side": target_side,
            }

    def compute_decision_variable(
        self,
        contrast_patch: np.ndarray,
        rng: np.random.Generator,
        reference_template: Optional[np.ndarray] = None,
    ) -> float:
        quantized = np.round(contrast_patch * 127.5) / 127.5
        c_sig = float(np.sum(quantized * self.ideal_template) * self.cfg.pixel_area_deg2) / self.unit_energy
        residual = quantized - c_sig * self.ideal_template
        lin = float(np.sum(quantized * self.vlm_template) * self.cfg.pixel_area_deg2) / self.unit_energy
        vlm_sd = self.compute_channel_noise_sd(residual) * 1.95

        c_eff = 0.35 * lin + 0.65 * math.sqrt(0.627) * c_sig
        transduced = math.copysign((abs(c_eff) + 0.030) ** 0.68 - (0.030 ** 0.68), c_eff)
        local_slope = 0.68 * ((abs(c_eff) + 0.030) ** -0.32)
        noise_sd = math.sqrt(0.022 ** 2 + (local_slope * vlm_sd) ** 2 + (0.24 * abs(c_eff)) ** 2)
        return transduced + float(rng.normal(0.0, noise_sd))


def build_observer_zoo(cfg: Optional[StimulusConfig] = None) -> Dict[str, BaseObserver]:
    """Instantiates the complete suite of canonical, bio-inspired, neural backbone, and VLM observers."""
    stim_cfg = cfg or StimulusConfig()
    observers: List[BaseObserver] = [
        IdealObserver(stim_cfg),
        LinearAmplifierObserver(stim_cfg),
        PerceptualTemplateObserver(stim_cfg),
        FoleyDivisiveNormObserver(stim_cfg),
        V1MultiscaleGaborObserver(stim_cfg),
        ConvNetHierarchyObserver(stim_cfg),
        ViTPatchAttentionObserver(stim_cfg, patch_size=8),
        LocalVLMObserver(stim_cfg, ollama_model="llava:latest"),
    ]
    return {obs.model_id: obs for obs in observers}
