#!/usr/bin/env python3
"""
stimuli.py - Calibrated Psychophysical Stimulus & Noise Generator
Generates 2D luminance arrays and composite VLM images for:
1. Equivalent Noise (Gabor / DoG in calibrated white Gaussian noise spectral density N_ext)
2. Classification Images (Reverse correlation trial pairs with known noise fields)
3. Critical Bandwidth (Notched-noise and narrowband spatial-frequency masking in 2D Fourier domain)
4. TvC Dipper Functions (Pedestal C_ped + Increment Delta_C)
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class StimulusConfig:
    """Spatial and photometric calibration parameters matching the Surface Pro Suite."""
    size_px: int = 64               # Patch resolution (64x64 px = 2.0 deg x 2.0 deg at 32 px/deg)
    px_per_deg: float = 32.0        # Calibrated pixels per degree of visual angle
    bg_luminance: float = 0.5       # Normalized mean luminance (0.5 in [0, 1])
    sf_cpd: float = 4.0             # Default target spatial frequency (cycles/deg)
    sigma_deg: float = 0.35         # Gaussian envelope standard deviation (deg)
    orientation_deg: float = 0.0    # Vertical carrier = 0 deg
    phase_deg: float = 0.0          # Cosine phase (symmetric ON-center) = 0 deg

    @property
    def deg_per_px(self) -> float:
        return 1.0 / self.px_per_deg

    @property
    def pixel_area_deg2(self) -> float:
        """Area of a single pixel in square degrees of visual angle."""
        return (self.deg_per_px) ** 2


def make_coordinate_grid(cfg: StimulusConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (x_deg, y_deg, r2_deg) centered at 0."""
    n = cfg.size_px
    coords = (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * cfg.deg_per_px
    x_deg, y_deg = np.meshgrid(coords, coords)
    r2_deg = x_deg ** 2 + y_deg ** 2
    return x_deg, y_deg, r2_deg


def generate_gabor_template(
    cfg: StimulusConfig,
    sf_cpd: Optional[float] = None,
    phase_deg: Optional[float] = None,
    orientation_deg: Optional[float] = None,
    sigma_deg: Optional[float] = None,
) -> np.ndarray:
    """
    Generates a zero-mean unit-peak contrast template S(x, y) in [-1, 1].
    A stimulus at Michelson/peak contrast C is I(x,y) = L0 * (1 + C * S(x,y)).
    """
    sf = cfg.sf_cpd if sf_cpd is None else sf_cpd
    phase = math.radians(cfg.phase_deg if phase_deg is None else phase_deg)
    theta = math.radians(cfg.orientation_deg if orientation_deg is None else orientation_deg)
    sigma = cfg.sigma_deg if sigma_deg is None else sigma_deg

    x_deg, y_deg, r2_deg = make_coordinate_grid(cfg)
    x_rot = x_deg * math.cos(theta) + y_deg * math.sin(theta)

    envelope = np.exp(-r2_deg / (2.0 * sigma ** 2))
    carrier = np.cos(2.0 * math.pi * sf * x_rot + phase)
    template = envelope * carrier
    # Enforce exact zero DC mean to prevent mean-luminance artifacts
    template = template - np.mean(template)
    peak = np.max(np.abs(template))
    if peak > 1e-12:
        template = template / peak
    return template


def generate_dog_template(
    cfg: StimulusConfig,
    sigma_c_deg: float = 0.20,
    sigma_s_deg: float = 0.40,
    polarity: str = "ON",
) -> np.ndarray:
    """
    Generates a balanced Difference-of-Gaussians (DoG) template in [-1, 1].
    """
    _, _, r2_deg = make_coordinate_grid(cfg)
    balance = (sigma_c_deg / sigma_s_deg) ** 2
    g_c = np.exp(-r2_deg / (2.0 * sigma_c_deg ** 2))
    g_s = np.exp(-r2_deg / (2.0 * sigma_s_deg ** 2))
    dog = (g_c - balance * g_s) / (1.0 - balance)
    dog = dog - np.mean(dog)
    peak = np.max(np.abs(dog))
    if peak > 1e-12:
        dog = dog / peak
    if polarity.upper() == "OFF":
        dog = -dog
    return dog


def compute_signal_energy(template: np.ndarray, contrast: float, cfg: StimulusConfig) -> float:
    """
    Computes continuous signal energy E = C^2 * sum(S^2) * dx * dy (in contrast^2 * deg^2).
    """
    return float((contrast ** 2) * np.sum(template ** 2) * cfg.pixel_area_deg2)


def compute_unit_template_energy(template: np.ndarray, cfg: StimulusConfig) -> float:
    """
    Computes unit-contrast template energy E_1 = sum(S^2) * dx * dy (in deg^2).
    For an Ideal Observer in white noise with two-sided power spectral density N_ext:
    d'_ideal = sqrt(E / N_ext) = C * sqrt(E_1 / N_ext).
    """
    return float(np.sum(template ** 2) * cfg.pixel_area_deg2)


def generate_white_noise(
    cfg: StimulusConfig,
    n_ext_deg2: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generates zero-mean Gaussian white contrast noise with two-sided power spectral
    density N_ext (in contrast^2 * deg^2).
    Since N_ext = sigma_pixel^2 * pixel_area_deg2, the per-pixel contrast SD is:
    sigma_pixel = sqrt(N_ext / pixel_area_deg2).
    """
    if n_ext_deg2 <= 0.0:
        return np.zeros((cfg.size_px, cfg.size_px), dtype=np.float64)
    sigma_px = math.sqrt(n_ext_deg2 / cfg.pixel_area_deg2)
    return rng.normal(0.0, sigma_px, size=(cfg.size_px, cfg.size_px))


def generate_notched_noise(
    cfg: StimulusConfig,
    n_ext_deg2: float,
    center_sf_cpd: float,
    notch_width_octaves: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generates 2D bandpass/notched spatial frequency noise centered around center_sf_cpd.
    When notch_width_octaves == 0, returns full broadband white noise.
    When notch_width_octaves > 0, removes energy in [center_sf * 2^(-w/2), center_sf * 2^(+w/2)].
    """
    raw_noise = generate_white_noise(cfg, n_ext_deg2, rng)
    if notch_width_octaves <= 1e-6 or n_ext_deg2 <= 0.0:
        return raw_noise

    n = cfg.size_px
    freqs_cpd = np.fft.fftfreq(n, d=cfg.deg_per_px)
    fx, fy = np.meshgrid(freqs_cpd, freqs_cpd)
    f_radial = np.sqrt(fx ** 2 + fy ** 2)

    low_cut = center_sf_cpd * (2.0 ** (-notch_width_octaves / 2.0))
    high_cut = center_sf_cpd * (2.0 ** (notch_width_octaves / 2.0))

    # Smooth raised-cosine transitions at notch edges (0.1 octave skirt) to prevent ringing
    mask = np.ones_like(f_radial)
    in_notch = (f_radial >= low_cut) & (f_radial <= high_cut)
    mask[in_notch] = 0.0

    fft_noise = np.fft.fft2(raw_noise)
    filtered = np.real(np.fft.ifft2(fft_noise * mask))
    return filtered


def compose_luminance_patch(
    contrast_field: np.ndarray,
    cfg: StimulusConfig,
    clip: bool = True,
) -> np.ndarray:
    """
    Converts a contrast modulation field delta_C(x,y) into normalized luminance [0, 1]:
    L(x,y) = L0 * (1 + delta_C(x,y)).
    """
    lum = cfg.bg_luminance * (1.0 + contrast_field)
    if clip:
        lum = np.clip(lum, 0.0, 1.0)
    return lum


def render_2afc_composite_png_base64(
    patch_left: np.ndarray,
    patch_right: np.ndarray,
    upscale: int = 3,
) -> str:
    """
    Renders a side-by-side 2AFC composite image (Left = Option 1, Right = Option 2)
    encoded as a base64 PNG string for local Vision-Language Models (e.g., Ollama LLaVA).
    """
    h, w = patch_left.shape
    pad = 16
    canvas_w = w * 2 + pad * 3
    canvas_h = h + pad * 2 + 20

    canvas = np.full((canvas_h, canvas_w), 128, dtype=np.uint8)
    left_u8 = np.clip(np.round(patch_left * 255.0), 0, 255).astype(np.uint8)
    right_u8 = np.clip(np.round(patch_right * 255.0), 0, 255).astype(np.uint8)

    y0 = pad + 16
    x_left = pad
    x_right = pad * 2 + w

    canvas[y0:y0 + h, x_left:x_left + w] = left_u8
    canvas[y0:y0 + h, x_right:x_right + w] = right_u8

    img = Image.fromarray(canvas, mode="L").convert("RGB")
    if upscale > 1:
        img = img.resize((canvas_w * upscale, canvas_h * upscale), resample=Image.Resampling.NEAREST)

    draw = ImageDraw.Draw(img)
    # Draw subtle framing boxes and labels "1" and "2"
    s = upscale
    draw.rectangle(
        [x_left * s - 2, y0 * s - 2, (x_left + w) * s + 1, (y0 + h) * s + 1],
        outline=(0, 188, 212),
        width=2,
    )
    draw.rectangle(
        [x_right * s - 2, y0 * s - 2, (x_right + w) * s + 1, (y0 + h) * s + 1],
        outline=(255, 179, 0),
        width=2,
    )
    draw.text(((x_left + w // 2 - 4) * s, 4 * s), "1", fill=(0, 230, 255))
    draw.text(((x_right + w // 2 - 4) * s, 4 * s), "2", fill=(255, 200, 0))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
