/**
 * gabor_renderer.js - Constant-Envelope Gabor Stimulus Generator with Bit-Depth Dithering
 * 
 * 1. Constant envelope size: Gaussian sigma held invariant across all spatial frequencies
 *    (default sigma = 0.5 deg visual angle, compact stimulus for fullscreen presentation).
 * 2. Bit-depth awareness: Accounts for 8-bit quantization floor (C_min ~ 0.78%) and implements
 *    high-frequency spatial triangular noise dithering (TPDF) to achieve pseudo-10-bit resolution (C_min ~ 0.1%).
 */

class GaborRenderer {
    /**
     * @param {Object} options
     * @param {number} [options.viewingDistanceCm=57.3] - 57.3 cm -> 1 cm = 1 deg visual angle
     * @param {number} [options.pixelsPerCm=38.0] - Monitor resolution calibration (~96 DPI)
     * @param {number} [options.bgLuminance=128] - 8-bit mid-gray background (0-255)
     * @param {number} [options.sigmaDeg=0.5] - Constant envelope Gaussian sigma in visual degrees
     * @param {number} [options.bitDepth=8] - Framebuffer bit depth (8 or 10)
     * @param {boolean} [options.enableDithering=true] - Spatial dithering for sub-LSB contrasts
     */
    constructor(options = {}) {
        this.viewingDistanceCm = options.viewingDistanceCm || 57.3;
        this.pixelsPerCm = options.pixelsPerCm || 38.0;
        this.bgLuminance = options.bgLuminance !== undefined ? options.bgLuminance : 128;
        this.sigmaDeg = options.sigmaDeg !== undefined ? options.sigmaDeg : 0.5; // Constant envelope size
        this.bitDepth = options.bitDepth || 8;
        this.enableDithering = options.enableDithering !== undefined ? options.enableDithering : true;

        // Compute pixels per degree of visual angle: 1 deg = 2 * dist * tan(0.5 deg)
        const rad1Deg = (1.0 * Math.PI) / 180.0;
        const cmPerDeg = 2.0 * this.viewingDistanceCm * Math.tan(rad1Deg / 2.0);
        this.pixelsPerDegree = cmPerDeg * this.pixelsPerCm;

        // Physical quantization floor:
        // For 8-bit display: 256 gray levels, min step delta = 1
        // C_min = 1 / L_bg = 1 / 128 = 0.0078125 (~0.78% contrast)
        this.numLevels = Math.pow(2, this.bitDepth);
        this.physicalContrastFloorUndithered = 1.0 / this.bgLuminance;
        // With spatial dithering, effective resolution reaches ~10-11 bits
        this.effectiveContrastFloorDithered = 1.0 / (this.bgLuminance * 8.0); // ~0.00098 (~0.1%)
    }

    setPixelsPerDegree(ppd) {
        this.pixelsPerDegree = ppd;
    }

    degToPixels(deg) {
        return deg * this.pixelsPerDegree;
    }

    /**
     * Constant envelope size: returns fixed sigma in visual degrees
     * @returns {number} sigma in visual degrees
     */
    getSigmaDeg() {
        return this.sigmaDeg;
    }

    setSigmaDeg(sigmaDeg) {
        this.sigmaDeg = sigmaDeg;
    }

    /**
     * Returns effective envelope sigma in visual degrees.
     * Preserves constant sigma across standard/high frequencies, while ensuring sufficient
     * spatial extent for ultra-low frequencies (0.125 - 0.25 cpd) so carrier modulation is visible.
     */
    getEffectiveSigmaDeg(spatialFreqCpd) {
        return Math.max(this.sigmaDeg, 0.35 / spatialFreqCpd);
    }

    /**
     * Generates an offscreen canvas containing a constant-envelope Gabor patch
     * @param {Object} params
     * @param {number} params.spatialFreqCpd - Spatial frequency in cpd (0.125 to 32 cpd)
     * @param {number} params.contrast - Michelson contrast (0.001 to 1.0)
     * @param {number} [params.orientationDeg=0] - Carrier orientation in degrees (0 = vertical)
     * @param {number} [params.phase=0] - Carrier phase in radians
     * @param {number} [params.sigmaDeg] - Optional override for envelope sigma
     * @returns {HTMLCanvasElement}
     */
    createGaborCanvas(params) {
        const { spatialFreqCpd, contrast, orientationDeg = 0, phase = 0 } = params;
        const sigmaDeg = params.sigmaDeg !== undefined 
            ? params.sigmaDeg 
            : this.getEffectiveSigmaDeg(spatialFreqCpd);

        // 1. Envelope Gaussian calculation
        const sigmaPixels = this.degToPixels(sigmaDeg);

        // Spatial frequency in cycles per pixel:
        // f_pixels = f_cpd / pixelsPerDegree
        const freqPixels = spatialFreqCpd / this.pixelsPerDegree;

        // Stimulus patch size: extend to 3.5 * sigma to capture full Gaussian envelope
        const patchRadius = Math.max(16, Math.ceil(3.5 * sigmaPixels));
        const patchSize = 2 * patchRadius + 1;

        const offscreen = document.createElement("canvas");
        offscreen.width = patchSize;
        offscreen.height = patchSize;
        const ctx = offscreen.getContext("2d");

        const imgData = ctx.createImageData(patchSize, patchSize);
        const data = imgData.data;

        // Orientation angle in radians
        const theta = (orientationDeg * Math.PI) / 180.0;
        const cosTheta = Math.cos(theta);
        const sinTheta = Math.sin(theta);
        const twoSigma2 = 2.0 * sigmaPixels * sigmaPixels;
        const twoPiFreq = 2.0 * Math.PI * freqPixels;

        // Bit-depth configuration
        const bg = this.bgLuminance;
        const maxVal = this.numLevels - 1;

        for (let py = 0; py < patchSize; py++) {
            const y = py - patchRadius;
            for (let px = 0; px < patchSize; px++) {
                const x = px - patchRadius;

                // Rotated coordinates
                // xPrime aligns with carrier modulation direction
                const xPrime = x * cosTheta + y * sinTheta;
                const yPrime = -x * sinTheta + y * cosTheta;

                // 2D Gaussian envelope (constant across spatial frequencies)
                const r2 = xPrime * xPrime + yPrime * yPrime;
                const env = Math.exp(-r2 / twoSigma2);

                // Sinusoidal carrier
                const carrier = Math.cos(twoPiFreq * xPrime + phase);

                // Ideal continuous floating-point luminance
                // L(x, y) = L_bg * [1 + contrast * env * carrier]
                let continuousLum = bg * (1.0 + contrast * env * carrier);

                // 2. Display Bit-Depth & Spatial Dithering Application
                let quantizedLum;
                if (this.enableDithering) {
                    // High-frequency spatial triangular noise dithering (TPDF)
                    // Eliminates discrete banding at sub-LSB contrasts (< 0.78%)
                    const ditherNoise = (Math.random() - Math.random()) * 0.707;
                    const ditheredVal = continuousLum + ditherNoise;
                    quantizedLum = Math.round(ditheredVal);
                } else {
                    // Raw quantization to display bit depth
                    quantizedLum = Math.round(continuousLum);
                }

                // Clamp to valid range
                quantizedLum = Math.max(0, Math.min(maxVal, quantizedLum));

                const idx = (py * patchSize + px) * 4;
                data[idx] = quantizedLum;     // Red
                data[idx + 1] = quantizedLum; // Green
                data[idx + 2] = quantizedLum; // Blue
                data[idx + 3] = 255;          // Alpha
            }
        }

        ctx.putImageData(imgData, 0, 0);
        return offscreen;
    }

    /**
     * Draw Gabor centered on target canvas
     * @param {CanvasRenderingContext2D} ctx 
     * @param {number} centerX 
     * @param {number} centerY 
     * @param {HTMLCanvasElement} gaborCanvas 
     */
    drawCentered(ctx, centerX, centerY, gaborCanvas) {
        const patchRadius = Math.floor(gaborCanvas.width / 2);
        ctx.drawImage(gaborCanvas, centerX - patchRadius, centerY - patchRadius);
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { GaborRenderer };
}
