# Vision Science Experiment: ON vs. OFF Difference of Gaussians (DoG) with QUEST+

This project implements a psychophysics experiment using **PsychoJS / HTML5 Canvas & WebGL** to measure human contrast discrimination thresholds between **ON-center** and **OFF-center Difference of Gaussians (DoG)** visual stimuli across spatial scales and retinal eccentricities.

Thresholds are estimated using the **multidimensional QUEST+ algorithm** ([Watson, 2017](https://doi.org/10.1167/17.3.10)).

---

## 1. Scientific Background

### Difference of Gaussians (DoG) Stimulus
In early visual processing (retinal ganglion cells and lateral geniculate nucleus), receptive fields exhibit antagonistic center-surround receptive field profiles modeled by a Difference of Gaussians:
$$DoG(r) = A \cdot \left[ \exp\left(-\frac{r^2}{2\sigma_c^2}\right) - \left(\frac{\sigma_c}{\sigma_s}\right)^2 \exp\left(-\frac{r^2}{2\sigma_s^2}\right) \right]$$

- **Center-to-Surround Ratio**: $\sigma_s = 2.0 \cdot \sigma_c$.
- **Integral Balance**: Zero net DC luminance shift above the neutral gray background.
- **Polarity**:
  - **ON-Center**: $A > 0$ (bright center, dark surround).
  - **OFF-Center**: $A < 0$ (dark center, bright surround).

### Factorial Design Matrix (12 Interleaved Conditions)
- **Factor 1: Spatial Scale (Size $\sigma_c$)**:
  - Small: $\sigma_c = 0.15^\circ$
  - Medium: $\sigma_c = 0.30^\circ$
  - Large: $\sigma_c = 0.60^\circ$
- **Factor 2: Retinal Eccentricity (Distance from Fixation)**:
  - $2.0^\circ, 4.0^\circ, 6.0^\circ, 8.0^\circ$ visual angle from central fixation.

---

## 2. QUEST+ Bayesian Psychometric Algorithm

Rather than classic fixed staircases, the experiment utilizes **QUEST+** ([Watson, 2017](https://doi.org/10.1167/17.3.10)):
1. **Psychometric Model**: Weibull psychometric function with chance level $\gamma = 0.5$ (2AFC), lapse rate $\lambda = 0.02$, slope $S = 2.5$, and candidate threshold space $T \in [-2.5, 0.0]$ in $\log_{10}$ contrast units (0.3% to 100% contrast).
2. **Entropy Minimization**: At each trial, the stimulus intensity $x^*$ is chosen to minimize expected posterior Shannon entropy:
   $$x^* = \arg\min_x \mathbb{E}[H(p(T \mid r, x))]$$
3. **Posterior Updating**: Bayes' rule updates the probability distribution over candidate thresholds immediately upon observing the participant's response.
4. **Interleaved QUEST+ Conditions**: All 12 conditions run concurrently in an interleaved sequence to prevent temporal expectation artifacts.

---

## 3. Project Structure

```
vision_experiment_dog_quest/
├── index.html                   # Experiment GUI & canvas runner
├── run_experiment.py            # Local web server runner (opens default browser)
├── simulate_experiment.py       # Simulation harness testing convergence on synthetic observer
├── css/
│   └── style.css                # Psychophysics dark-mode and touch layout
└── js/
    ├── experiment.js            # State machine & trial controller
    ├── questplus.js             # Full QUEST+ implementation (Watson, 2017)
    ├── dog_renderer.js          # Procedural Difference of Gaussians canvas generator
    └── data_exporter.js         # CSV & JSON data logger and downloader
```

---

## 4. How to Run

### Method 1: Local Python Server (Recommended)
In PowerShell or Command Prompt:
```powershell
uv run python vision_experiment_dog_quest/run_experiment.py
```
This automatically launches your browser (Edge on Surface Pro) at `http://127.0.0.1:8088/index.html`.

### Method 2: Direct Browser Launch
Simply double-click [`vision_experiment_dog_quest/index.html`](file:///c:/Users/chris_7wwspmu/OneDrive/Code_2026_OneDrive/vision_experiment_dog_quest/index.html).

---

## 5. Controls & Participant Task

1. Keep gaze fixed on the central **fixation cross**.
2. A brief ($200\text{ ms}$) target will flash at an eccentricity of $2^\circ, 4^\circ, 6^\circ,$ or $8^\circ$.
3. Indicate whether the stimulus center was bright (ON) or dark (OFF):
   - **Keyboard**: Press **F** or **Up Arrow** for ON; press **J** or **Down Arrow** for OFF.
   - **Surface Pro Touch Screen**: Tap the on-screen **ON** (white) or **OFF** (black) buttons.
4. Upon completing all conditions, an interactive psychometric curve is plotted, and you can download:
   - **Trial-by-trial logs CSV**: `sub-XXXX_dog_quest_trials.csv`
   - **Thresholds summary CSV**: `sub-XXXX_dog_quest_thresholds_summary.csv`
   - **Full session JSON**: `sub-XXXX_dog_quest_session.json`
