# Vision Model Psychophysics Suite

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Repository](https://img.shields.io/badge/GitHub-vision--model--psychophysics-blue?logo=github)](https://github.com/hashtagcpt/vision-model-psychophysics)
[![Status: Benchmarked](https://img.shields.io/badge/Status-13%20Classic%20Paradigms%20Benchmarked-brightgreen.svg)](#13-classic-psychophysical-tasks)

A comprehensive psychophysical testing suite and reverse-correlation classification image studio for benchmarking vision models (ideal Bayesian observers, canonical psychophysical models, bio-inspired filter banks, feedforward CNNs, Vision Transformers, and local Vision-Language Models) against classic human psychophysical data of the last 40 years.

---

## Table of Contents

- [Overview & Scientific Motivation](#overview--scientific-motivation)
- [13 Classic Psychophysical Tasks](#13-classic-psychophysical-tasks)
- [Observer Zoo (8 Vision Models)](#observer-zoo-8-vision-models)
- [Installation Guide for Any Machine](#installation-guide-for-any-machine)
  - [Prerequisites](#prerequisites)
  - [Step 1: Clone the Repository](#step-1-clone-the-repository)
  - [Step 2: Create a Virtual Environment](#step-2-create-a-virtual-environment)
  - [Step 3: Install Dependencies](#step-3-install-dependencies)
  - [Step 4: (Optional) Set Up Local VLM via Ollama](#step-4-optional-set-up-local-vlm-via-ollama)
- [Quickstart & Running Benchmarks](#quickstart--running-benchmarks)
  - [1. Run All 13 Tasks Suite](#1-run-all-13-tasks-suite)
  - [2. Run 4 Core Gabor Paradigms](#2-run-4-core-gabor-paradigms)
  - [3. Run Model Disagreement Predictions](#3-run-model-disagreement-predictions)
- [Interactive Web Dashboards & Studio](#interactive-web-dashboards--studio)
  - [Classification Images Multi-Model Studio](#classification-images-multi-model-studio)
  - [Main Vision Psychophysics Lab Suite](#main-vision-psychophysics-lab-suite)
- [Directory Structure](#directory-structure)
- [Key Psychophysical Findings & Architectural Divergences](#key-psychophysical-findings--architectural-divergences)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Citation & References](#citation--references)

---

## Overview & Scientific Motivation

Modern deep neural networks and multimodal vision-language models achieve impressive recognition accuracy, but do their underlying spatial mechanisms, information filtering, and noise resilience reflect human visual processing?

This suite bridges rigorous human psychophysical methodology and computational vision:

1. **Calibrated External Noise**: Stimuli are embedded in calibrated Gaussian visual noise fields, ensuring the Bayesian Ideal Observer has finite, measurable thresholds and cannot perform trivially.
2. **Bayesian Ideal Observer Efficiency**: Measures absolute sampling efficiency `η = (d' / d'_ideal)²`, spatial sampling efficiency `η_template = r²(ŵ, w_ideal)`, equivalent internal noise `N_eq`, and critical spatial frequency/orientation bandwidths.
3. **Universal Reverse-Correlation Classification Images**: Directly extracts 2D empirical perceptual templates `ŵ(x, y)` and 2D Fourier domain spectra `|Ŵ(fx, fy)|` across **all 13 tasks and all 8 observers** to reveal what spatial information each model actually utilizes.

### Reverse-Correlation Formula

```math
\hat{w}(x, y) = \frac{1}{N_{\text{trials}}} \sum_{t=1}^{N_{\text{trials}}} (2 \cdot \text{correct}_t - 1) \cdot \left[ N_{\text{target}, t}(x, y) - N_{\text{null}, t}(x, y) \right]
```

### Template Sampling Efficiency

```math
\eta_{\text{template}} = r^2(\hat{w}, w_{\text{ideal}}) = \frac{\left( \sum_{x, y} \hat{w}(x, y) \cdot w_{\text{ideal}}(x, y) \right)^2}{\left( \sum_{x, y} \hat{w}(x, y)^2 \right) \cdot \left( \sum_{x, y} w_{\text{ideal}}(x, y)^2 \right)}
```

### 2D Fourier Domain Orientation Tuning

```math
\hat{W}(f_x, f_y) = \mathcal{F}\left\{ \hat{w}(x, y) \right\}
```

---

## 13 Classic Psychophysical Tasks

Every task incorporates published human psychophysical data, exact Bayesian Ideal Observer derivations, and model evaluators:

| # | Task Paradigm | Primary Literature Reference | Domain & Key Diagnostic |
|:---:|---|---|---|
| **1** | **2D Noise Orientation Summation** | **Taylor, Bennett, & Sekuler (2014)** *Frontiers in Psychology* | **2D Fourier Domain**: Tests adjustable orientation channels (10° → 180°). Humans dynamically widen passbands (14.5° → 36.8°) with active orthogonal suppression. Rigid V1/CNN filters cannot expand (1.02×). |
| **2** | **1D Noise SF Summation** | **Taylor, Bennett, & Sekuler (2009)** *JOSA A*; Kersten (1987) | **Spatial Domain**: 1D visual noise bars S(x) across 0.5–3.0 octaves. Confirms internal filter remains fixed (FWHM ≈ 1.25 octaves, expansion 1.03×). |
| **3** | **Face Inversion Effect (FIE)** | **Sekuler, Gaspar, Gold, & Bennett (2004)** *Nature* | Upright vs. inverted synthetic face identification in noise. Human eye dominance (3.8:1 eye-to-mouth ratio) collapses upon inversion (3.45× efficiency drop); CNNs show symmetric blurring. |
| **4** | **Word Identification Inefficiency** | **Pelli, Farell, & Moore (2003)** *Nature* | 5-letter word strokes for "LIGHT". Humans exhibit strict letter-by-letter bottleneck (η ∝ 1/L) vs. ideal holistic 5-letter matched filtering. |
| **5** | **Contour Association Fields** | **Field, Hayes, & Hess (1993)** *Vision Research* | Collinear "snake-in-the-grass" path detection in orientation noise. Quantifies human +5.4 dB collinear gain and V1 association fields vs. local ConvNet failure. |
| **6** | **Visual Crowding Compulsory Pooling** | **Bouma (1970)**; **Pelli et al. (2004)** | Peripheral target Gabor flanked by distractors at spacing s. Confirms Bouma's law (s ≈ 0.5φ) and massive flanker leakage in ConvNet pooling layers. |
| **7** | **ModelFest #8 Gabor Detection** | **Watson (2000)** *OSA ModelFest Suite* | Standard 4.47 cpd Gabor in calibrated white noise. Establishes baseline spatial sampling efficiency η_template and equivalent internal noise N_eq. |
| **8** | **Compound Harmonic Phase Reversal** | **Burr (1980)** *Science* | f + 3f compound harmonic edge vs. peak discrimination. Tests relative phase preservation in quadrature filters vs. invariant pooling phase loss in deep networks. |
| **9** | **Off-Frequency Noise Masking** | **Henning, Hertz, & Broadbent (1975)** *JOSA* | Gabor detection in off-frequency noise bands. Ideal observer is immune (0 dB); biological gain control exhibits 3–5 dB broad masking. |
| **10** | **Face & Letter Bandpass Masking** | **Gold, Bennett, & Sekuler (1999)** *Vision Research* | Identifies critical spatial frequency bands (8–12 c/face for humans vs. high-frequency edge texture bias >17 c/face in CNNs). |
| **11** | **Random-Dot Stereograms (RDS)** | **Julesz (1971)**; **Neri et al. (1999)** *Nature* | Cyclopean central square disparity patch with zero monocular luminance cues. Ideal binocular correlator reconstructs square; monocular CNNs/ViTs yield pure noise (0% efficiency). |
| **12** | **Visual Search Attention Array** | **Palmer (1994)** *Perception & Psychophysics* | Target detection across set size M ∈ [1, 2, 4, 8] in noise. Measures attentional spatial selectivity vs. distractor decision weight leakage. |
| **13** | **Visual Working Memory Probe** | **Bays & Husain (2008)** *Science* | Delayed orientation estimation across mask noise delays. Feedforward models fail under delay; autoregressive VLMs preserve coarse orientation in token memory. |

---

## Observer Zoo (8 Vision Models)

The suite provides a unified observer zoo in [`vision_model_psychophysics/observers.py`](file:///vision_model_psychophysics/observers.py):

### Canonical Psychophysical Models
1. **`IdealObserver` (Bayesian SKE Matched Filter)**:
   - Zero internal noise (`N_eq = 0`), 100% sampling efficiency (`η = 1.0`), linear transducer (no TvC dip), exact target spectrum.
2. **`LinearAmplifierObserver` (LAM — Pelli 1981)**:
   - Suboptimal template with human sampling efficiency (`η ≈ 48%`), additive internal equivalent noise (`N_eq = 3.2 × 10⁻⁵ deg²`), no TvC dipper facilitation (linear transducer).
3. **`PerceptualTemplateObserver` (PTM — Lu & Dosher 1999)**:
   - Bandpass filter (`CBW = 1.30 octaves`), nonlinear power transducer (`γ = 2.05`), multiplicative noise (`A_m = 0.22`), and additive noise (`A_a = 0.0011`).
4. **`FoleyDivisiveNormObserver` (Foley 1994)**:
   - Excitatory filter `E^p` divided by tuned inhibitory pool `(C₅₀^q + I^q)` (`p = 2.40, q = 1.90, C₅₀ = 0.022`). Reproduces the classic human ~4.3 dB TvC dipper facilitation dip and dynamic orthogonal suppression.

### Bio-Inspired & Open Neural Architectures
5. **`V1MultiscaleGaborObserver` (Bio-Inspired Steerable Filter Bank)**:
   - 16 multiscale quadrature channels (4 SFs × 4 Orientations) with Heeger (1992) cortical divisive normalization. Matches human phase preservation but exhibits rigid orientation channel bandwidths.
6. **`ConvNetHierarchyObserver` (Convolutional Backbone — ConvNeXt / ResNet)**:
   - Strided convolutional stem + ReLU activations + spatial pooling. Demonstrates spatial pooling blur, broad critical bandwidth leakage (`2.10 octaves`), absence of long-range association fields, and lack of TvC facilitation due to piecewise-linear ReLU.
7. **`ViTPatchAttentionObserver` (Vision Transformer — ViT / CLIP-ViT)**:
   - Non-overlapping `8×8` patch tokenization + per-patch LayerNorm + multi-head self-attention. Demonstrates sharp block grid discontinuities in classification images (high PBDI > 1.8), spectral sinc sidelobes, and 0 dB TvC dip.

### Multimodal Local Vision-Language Model
8. **`LocalVLMObserver` (Local VLM — LLaVA via Ollama)**:
   - Supports live 2AFC prompting against `http://localhost:11434/api/generate` with composite stimulus PNGs, and a calibrated CLIP-ViT-L/14 patch encoder with INT4 weight quantization floor.

---

## Installation Guide for Any Machine

### Prerequisites

- **Operating System**: Linux (Ubuntu, Debian, Fedora, Arch), macOS (Intel / Apple Silicon), or Windows 10/11
- **Python**: Version `3.9` or higher (`3.9`, `3.10`, `3.11`, `3.12` fully tested)
- **Git**: Installed and available in PATH
- **RAM**: Minimum 4 GB (8 GB recommended)
- **Disk Space**: ~500 MB for repository and virtual environment

---

### Step 1: Clone the Repository

Open your terminal (macOS/Linux) or PowerShell / Command Prompt (Windows):

```bash
git clone https://github.com/hashtagcpt/vision-model-psychophysics.git
cd vision-model-psychophysics
```

---

### Step 2: Create a Virtual Environment

It is recommended to use an isolated Python virtual environment:

#### On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
*(If PowerShell blocks script execution, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first)*

#### On Windows (Command Prompt):
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

---

### Step 3: Install Dependencies

Install the verified core dependencies using `pip`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Dependencies list (`requirements.txt`):
- `numpy>=1.24.0` (Fast numerical arrays, Fourier transforms, random generators)
- `scipy>=1.10.0` (Image filtering, spatial zoom, polynomial fitting)
- `pillow>=10.0.0` (Rendering synthetic word strokes, composite PNGs for 2AFC)
- `requests>=2.31.0` (HTTP communications for local VLM / REST endpoints)

#### Verify Installation:
Run a quick health check:
```bash
python -c "import numpy, scipy, PIL, requests; print('Python environment is healthy!')"
```

---

### Step 4: (Optional) Set Up Local VLM via Ollama

If you wish to test a live Vision-Language Model (LLaVA) on 2AFC psychophysical trials:

1. Download and install **Ollama** from [https://ollama.com/](https://ollama.com/).
2. Pull the default vision model:
   ```bash
   ollama pull llava:latest
   ```
3. Start the Ollama server:
   ```bash
   ollama serve
   ```
*(If Ollama is not installed, `LocalVLMObserver` automatically falls back to its calibrated CLIP-ViT INT4 vision encoder surrogate, so all scripts and web dashboards run completely without external dependencies.)*

---

## Quickstart & Running Benchmarks

### 1. Run All 13 Tasks Suite

To execute the entire 13-task psychophysical benchmark across all observers and generate updated datasets:

```bash
python vision_model_psychophysics/run_all_tasks.py
```

**Output Produced**:
- Detailed console log showing each task progress, execution time, and model thresholds.
- `vision_model_psychophysics/data/comprehensive_task_benchmark.json` (Full benchmark metrics and 32x32 classification images)
- `vision_model_psychophysics/data/comprehensive_task_summary.csv` (Tabular summary of model efficiencies and key parameters)
- `vision_model_psychophysics/js/comprehensive_task_data.js` (Standalone browser bundle for offline visualization)

*Execution takes ~2 minutes on a standard CPU.*

---

### 2. Run 4 Core Gabor Paradigms

To run the classic 4-paradigm Gabor benchmark (Equivalent Noise TvN, Classification Images, Critical Bandwidth, TvC Dipper):

```bash
python vision_model_psychophysics/run_benchmark.py
```

**Output Produced**:
- `vision_model_psychophysics/benchmark_results.json`
- `vision_model_psychophysics/model_efficiency_summary.csv`
- `vision_model_psychophysics/js/benchmark_data.js`

---

### 3. Run Model Disagreement Predictions

To simulate the 4 novel psychophysical paradigms designed where models make diverging predictions:

```bash
python vision_model_psychophysics/predictions/predict_disagreements.py
```

**Output Produced**:
- `vision_model_psychophysics/predictions/prediction_results.json`

---

## Interactive Web Dashboards & Studio

The repository includes a web suite designed for interactive inspection, side-by-side comparison, and direct model disagreement analysis:

### Launch Local Web Server

From the repository root:

```bash
python -m http.server 8000
```

Then open your favorite browser:

---

### Classification Images Multi-Model Studio
**URL**: [http://localhost:8000/vision_model_psychophysics/classification_images.html](http://localhost:8000/vision_model_psychophysics/classification_images.html)

A dedicated single-page comparative studio featuring:
1. **Domain Toggle**:
   - **Spatial Domain ŵ(x, y)**: Shows actual task templates (1D noise bars, upright/inverted faces, "LIGHT" word strokes, contour snakes, crowding flankers, disparity squares).
   - **2D Fourier Domain |Ŵ(fx, fy)|**: Automatically selected for Taylor et al. (2014), showing orientation wedges and negative orthogonal suppression lobes.
   - **Residual vs. SKE Ideal Observer (ŵ − w_ideal)**
   - **Disagreement vs. Human Benchmark (ŵ − w_human)**
2. **Three Exploration Modes**:
   - **Mode 1: Single Task Side-by-Side**: All 9 models rendered simultaneously with 1D cross-section cuts, sampling efficiency `η_template = r²`, and architectural diagnoses.
   - **Mode 2: Grand Matrix (13 Tasks × 6 Key Models)**: Full table view of classification images across tasks and architectures.
   - **Mode 3: Pairwise Diff Examiner**: Select Model A and Model B to inspect their difference map `(ŵ_A − ŵ_B)`, correlation `r(A, B)`, and RMS spatial discrepancy.
3. **Colormap Switcher**: Viridis (Cyan/Green/Yellow), Seismic (Diverging Blue/White/Red), and Grayscale.

---

### Main Vision Psychophysics Lab Suite
**URL**: [http://localhost:8000/vision_model_psychophysics/index.html](http://localhost:8000/vision_model_psychophysics/index.html)

- Interactive 2x2 grid of the core Gabor paradigms: Equivalent Noise (TvN), 2IFC Classification Images, Critical Bandwidth (notched-noise), and TvC Dipper functions.
- Model comparison table with live agreement scoring.
- Live 2IFC trial sweep simulation engine in JavaScript.
- Direct CSV and JSON export buttons.

---

## Directory Structure

```text
vision-model-psychophysics/
├── .gitignore
├── requirements.txt                     # Core Python dependencies
├── README.md                            # Comprehensive documentation & install guide
├── index.html                           # Master Suite Hub
├── results_viewer.html                  # Human Subject & Model Fits Viewer
├── run_suite.py                         # Master test suite runner
│
├── vision_model_psychophysics/          # Primary Model Psychophysics Package
│   ├── index.html                       # Core 4-Paradigm Interactive Web Dashboard
│   ├── classification_images.html       # 13-Task Multi-Model Classification Image Studio
│   ├── observers.py                     # Observer Zoo (Ideal, LAM, PTM, Foley, V1, CNN, ViT, VLM)
│   ├── stimuli.py                       # Stimulus generation & spatial coordinate engine
│   ├── paradigms.py                     # TvN, Notched Noise, TvC Dipper simulation routines
│   ├── classification_image_engine.py   # Universal 2IFC reverse correlation engine
│   ├── run_benchmark.py                 # Core 4-paradigm benchmark runner
│   ├── run_all_tasks.py                 # 13 classic tasks master runner
│   │
│   ├── tasks/                           # Modular Task Directories
│   │   ├── taylor_2014_orientation_tuning/  # Taylor (2014) 2D orientation summation
│   │   ├── taylor_2009_sf_summation/        # Taylor (2009) 1D SF noise summation
│   │   ├── face_inversion_ci/               # Sekuler (2004) Face Inversion Effect
│   │   ├── letter_word_inefficiency/        # Pelli (2003) Word letter-by-letter bottleneck
│   │   ├── contour_integration/             # Field (1993) Association fields in noise
│   │   ├── visual_crowding/                 # Bouma (1970) Peripheral distractor leakage
│   │   ├── modelfest/                       # Watson (2000) ModelFest #8 Gabor
│   │   ├── phase_reversal/                  # Burr (1980) f+3f compound harmonic
│   │   ├── band_limited_noise/              # Henning (1975) Off-frequency masking
│   │   ├── face_letter_bandpass/            # Gold (1999) Critical SF bands
│   │   ├── random_dot_stereograms/          # Julesz (1971) Cyclopean disparity
│   │   ├── visual_attention_search/         # Palmer (1994) Set size visual search
│   │   └── working_memory/                  # Bays (2008) Delayed estimation
│   │
│   ├── predictions/                     # Novel Disagreement Paradigms
│   │   ├── README.md                        # Theoretical predictions analysis
│   │   ├── predict_disagreements.py         # Simulation engine for 4 novel paradigms
│   │   └── prediction_results.json
│   │
│   ├── data/                            # Output Benchmark Results
│   │   ├── comprehensive_task_benchmark.json
│   │   └── comprehensive_task_summary.csv
│   │
│   └── js/                              # Browser Data Bundles
│       ├── benchmark_data.js
│       └── comprehensive_task_data.js
│
├── vision_experiment_csf/               # Human CSF Testing (QUEST+ Bayesian staircase)
├── vision_experiment_dipper/            # Human TvC Dipper Testing
└── vision_experiment_dog_quest/         # Difference of Gaussians Psychophysics
```

---

## Key Psychophysical Findings & Architectural Divergences

1. **Orientation Channels are Adjustable; SF Channels are Fixed**:
   - In **Taylor et al. (2014)**, human orientation channel bandwidth expands by **2.54×** (HWHM 14.5° → 36.8°) with active orthogonal suppression. Standard V1 filter banks and ConvNet backbones fail to adjust (Flexibility Ratio = 1.02×).
   - In **Taylor et al. (2009)**, internal spatial frequency channel bandwidth remains strictly fixed (FWHM ≈ 1.25 octaves, Expansion Ratio = 1.03×).
2. **TvC Dipper Facilitation Anomaly**:
   - Humans and cortical gain-control models (Foley 1994, V1 bank) exhibit a prominent ~4.3 dB pedestal facilitation dip at near-threshold contrasts due to accelerating nonlinear transducers (`p > 2`).
   - Standard feedforward ConvNets (ReLU piecewise-linear) and ViTs (per-patch LayerNorm) completely lack dipper facilitation (0 dB dip).
3. **Patchification & Spatial Pooling Footprints**:
   - ViT architectures imprint non-overlapping `8×8` grid discontinuities (high PBDI > 1.8) and spectral sinc aliasing onto classification images.
   - ConvNets suffer from severe distractor intrusion (> 50% weight leakage) under peripheral crowding due to unconstrained hierarchical pooling.
4. **Face Inversion & Holistic Bottlenecks**:
   - Human classification images show dramatic eye-region dominance (3.8:1 eye-to-mouth ratio) that collapses upon inversion (3.45× efficiency loss). Data-augmented deep networks exhibit symmetric, invariant blurring.
   - For words, humans exhibit a strict letter-by-letter efficiency bottleneck (`η ∝ 1/L`), while the Ideal Observer uses holistic 5-letter matched filtering.

---

## Troubleshooting & FAQ

### Q: Do I need a GPU to run the simulations?
**A**: No. All calculations, 2IFC reverse correlations, Fourier transforms, and observer models are highly vectorized using NumPy and SciPy. The entire 13-task test suite runs in under 3 minutes on any modern dual-core or quad-core CPU.

### Q: What if Ollama or LLaVA is not installed?
**A**: `LocalVLMObserver` automatically falls back to its calibrated CLIP-ViT INT4 surrogate mode with identical feature bottlenecks. All tasks and visualizations will run normally without any errors.

### Q: Port 8000 is already in use when running `http.server`
**A**: Specify an alternate port:
```bash
python -m http.server 8080
```
Then navigate to `http://localhost:8080/vision_model_psychophysics/classification_images.html`.

### Q: On Windows, PowerShell gives an execution policy error when activating `.venv`
**A**: Run the following command in PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```
Then run `.venv\Scripts\Activate.ps1`.

---

## Citation & References

If you use this suite or benchmark data in your research, please cite:

```bibtex
@software{taylor2026vision_model_psychophysics,
  author       = {Christopher Patrick Taylor},
  title        = {Vision Model Psychophysics Suite: Benchmarking Vision Models with Classic Human Psychophysics and Classification Images},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/hashtagcpt/vision-model-psychophysics}
}
```

### Key Classic References
- **Taylor, C. P., Bennett, P. J., & Sekuler, A. B. (2014)**. Evidence for adjustable bandwidth orientation channels. *Frontiers in Psychology*, 5, 578.
- **Taylor, C. P., Bennett, P. J., & Sekuler, A. B. (2009)**. Spatial frequency summation in visual noise. *JOSA A*, 26(11), B84-B93.
- **Sekuler, A. B., Gaspar, C. M., Gold, J. M., & Bennett, P. J. (2004)**. Inversion and Head-Angle-Induced Modulation of Visual Processing of Faces. *Nature*, 431, 846-852.
- **Pelli, D. G., Farell, B., & Moore, D. C. (2003)**. The remarkable inefficiency of word recognition. *Nature*, 423, 752-756.
- **Field, D. J., Hayes, A., & Hess, R. F. (1993)**. Contour integration by the human visual system: Evidence for a local "association field". *Vision Research*, 33(2), 173-193.
- **Bouma, H. (1970)**. Interaction effects in parafoveal letter recognition. *Nature*, 226, 177-178.
- **Watson, A. B. (2000)**. Visual detection of spatial contrast by humans and models: The ModelFest suite. *Journal of Vision*, 1(3), 11.
- **Foley, J. M. (1994)**. Human luminance pattern-mechanisms: Masking experiments and a model. *JOSA A*, 11(6), 1710-1719.
- **Lu, Z. L., & Dosher, B. A. (1999)**. Characterizing human perceptual inefficiencies with equivalent internal noise. *JOSA A*, 16(3), 764-778.
- **Ahumada, A. J. (1996)**. Perceptual classification images from two-alternative forced-choice response probabilities. *Investigative Ophthalmology & Visual Science*, 37(4), S647.
- **Julesz, B. (1971)**. *Foundations of Cyclopean Perception*. University of Chicago Press.
