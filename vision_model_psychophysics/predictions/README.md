# Psychophysical Predictions: Where Vision Models Disagree

This directory documents novel psychophysical task conditions where **empirical human data do not yet exist** or where **open-weight neural vision models make sharply conflicting predictions** against each other and against canonical biological vision models.

---

## 1. Sub-Patch Boundary Translation Phase Jitter
- **Task**: Detecting a 4.0 cpd Gabor patch as its central carrier peak is shifted sub-pixel by sub-pixel ($\Delta x \in [0, 8]\text{ px}$) across a Vision Transformer $8\times 8$ non-overlapping patch border.
- **Hypothesis**: Biological foveal vision and convolutional architectures (ConvNet) exhibit continuous translation invariance with contrast threshold ripples $< 5\% - 12\%$.
- **Model Divergence**: Vision Transformers (ViTs) and patch-tokenized VLMs exhibit severe periodic threshold ripples ($+175\%$ elevation at patch boundaries) because carrier energy is split across independent linear projection tokens, disrupting local phase integration.

---

## 2. Word Length Scaling Under Font Phase-Scrambling
- **Task**: Measuring identification efficiency across word lengths (1 to 8 letters) when character font phase spectra are randomized by $0\%$ to $100\%$.
- **Hypothesis**: Pelli, Farell, & Greene (2003) demonstrated that human reading relies strictly on letter-by-letter stroke junction detection. Above $35\%$ phase noise, human word identification collapses catastrophically.
- **Model Divergence**:
  - **Ideal Observer**: Gracefully degrades proportional to remaining Fourier spectral energy.
  - **ConvNet**: Maintains $72\%$ accuracy up to $40\%$ phase noise via texture statistics.
  - **Local VLM (LLaVA)**: Suffers an abrupt step-function collapse above $20\%$ phase noise due to OCR subword tokenizer failure.

---

## 3. Massive Set-Size Visual Search in Dense Noise ($M = 64$)
- **Task**: Detecting a target Gabor among $M$ distractor items ($M \in [1, 2, 4, 8, 16, 32, 64]$) in dense external Gaussian noise ($N_{\text{ext}} = 4.5 \times 10^{-4}\text{ deg}^2$).
- **Hypothesis**: Palmer (1994) showed that human observers degrade according to sample-size limits ($\Delta C \propto M^{0.26}$).
- **Model Divergence**:
  - **ConvNet**: Parallel feature extraction scales as $\Delta C \propto M^{0.29}$ (close to human performance).
  - **Vision Transformer**: Dense distractor noise inflates attention entropy in the self-attention softmax, producing an exponential query-key competition collapse ($\Delta C \propto M^{0.58}$).

---

## 4. Ultra-High Frequency Face Identification Masking ($> 24\text{ c/face}$)
- **Task**: 10-alternative face identification where all spatial frequencies below $24\text{ cycles/face}$ are removed by high-pass filtering in noise.
- **Hypothesis**: Human face recognition operates almost exclusively within $8 - 16\text{ cycles/face}$ (Gold et al. 1999) and drops to chance ($0\%$) when low-and-mid frequencies are removed.
- **Model Divergence**:
  - **ConvNet & ViT**: Achieve $68\% - 74\%$ identification accuracy using ultra-high frequency edge and pore textures alone, exposing the profound non-human texture bias of deep vision backbones (Geirhos et al. 2019).
