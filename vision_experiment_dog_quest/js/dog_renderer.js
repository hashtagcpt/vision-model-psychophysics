/**
 * dog_renderer.js - Difference of Gaussians (DoG) Procedural Stimulus Renderer
 * 
 * Generates physiologically grounded ON-center and OFF-center DoG luminance profiles:
 * DoG(r) = A * [ exp(-r^2 / (2 * sigma_c^2)) - (sigma_c / sigma_s)^2 * exp(-r^2 / (2 * sigma_s^2)) ]
 * where sigma_s = 2.0 * sigma_c (DC-balanced).
 */

class DogRenderer {
    /**
     * @param {Object} options
     * @param {number} [options.viewingDistanceCm=57.3] - 57.3 cm ensures 1 cm ~ 1 deg visual angle
     * @param {number} [options.pixelsPerCm=38.0] - Display PPI conversion (default ~96 DPI screen)
     * @param {number} [options.bgLuminance=128] - Neutral mid-gray background (0-255)
     */
    constructor(options = {}) {
        this.viewingDistanceCm = options.viewingDistanceCm || 57.3;
        this.pixelsPerCm = options.pixelsPerCm || 38.0;
        this.bgLuminance = options.bgLuminance !== undefined ? options.bgLuminance : 128;
        
        // Compute pixels per degree of visual angle
        // 1 deg = 2 * viewingDistance * tan(0.5 deg)
        const rad1Deg = (1.0 * Math.PI) / 180.0;
        const cmPerDeg = 2.0 * this.viewingDistanceCm * Math.tan(rad1Deg / 2.0);
        this.pixelsPerDegree = cmPerDeg * this.pixelsPerCm;
    }

    /**
     * Set calibrated display parameters
     * @param {number} pixelsPerDeg 
     */
    setPixelsPerDegree(pixelsPerDeg) {
        this.pixelsPerDegree = pixelsPerDeg;
    }

    /**
     * Convert degrees of visual angle to screen pixels
     * @param {number} deg 
     * @returns {number}
     */
    degToPixels(deg) {
        return deg * this.pixelsPerDegree;
    }

    /**
     * Generate an offscreen canvas containing the DoG stimulus
     * @param {Object} params
     * @param {number} params.sigmaCenterDeg - Center standard deviation in visual degrees (e.g. 0.15, 0.3, 0.6)
     * @param {number} params.contrast - Michelson contrast (0.005 to 1.0)
     * @param {string} params.polarity - "ON" (bright center) or "OFF" (dark center)
     * @returns {HTMLCanvasElement}
     */
    createDogCanvas(params) {
        const { sigmaCenterDeg, contrast, polarity } = params;
        const sigmaC = this.degToPixels(sigmaCenterDeg);
        const sigmaS = 2.0 * sigmaC;
        
        // Stimulus patch size: extend to 3.5 * sigma_surround to capture full surround tail
        const patchRadius = Math.ceil(3.5 * sigmaS);
        const patchSize = 2 * patchRadius + 1;

        const offscreen = document.createElement("canvas");
        offscreen.width = patchSize;
        offscreen.height = patchSize;
        const ctx = offscreen.getContext("2d");

        const imgData = ctx.createImageData(patchSize, patchSize);
        const data = imgData.data;

        // Balance ratio: (sigmaC / sigmaS)^2 ensures zero DC offset (equal area under center and surround)
        const balanceRatio = Math.pow(sigmaC / sigmaS, 2.0);

        // Amplitude sign: ON-center > 0, OFF-center < 0
        const sign = polarity === "ON" ? 1.0 : -1.0;
        
        // Maximum peak DoG value at r = 0 is: 1.0 - balanceRatio
        const peakValAtZero = 1.0 - balanceRatio;
        const normFactor = peakValAtZero > 0 ? (1.0 / peakValAtZero) : 1.0;

        const twoSigmaC2 = 2.0 * sigmaC * sigmaC;
        const twoSigmaS2 = 2.0 * sigmaS * sigmaS;

        for (let py = 0; py < patchSize; py++) {
            const dy = py - patchRadius;
            const dy2 = dy * dy;
            for (let px = 0; px < patchSize; px++) {
                const dx = px - patchRadius;
                const r2 = dx * dx + dy2;

                // Center Gaussian and Surround Gaussian
                const gCenter = Math.exp(-r2 / twoSigmaC2);
                const gSurround = Math.exp(-r2 / twoSigmaS2);

                // Difference of Gaussians
                const rawDog = gCenter - balanceRatio * gSurround;
                const normalizedDog = rawDog * normFactor;

                // Modulate by peak contrast and polarity
                const contrastDelta = sign * contrast * normalizedDog;

                // Apply to neutral background luminance
                // Contrast delta ranges from -contrast to +contrast
                let lum = this.bgLuminance * (1.0 + contrastDelta);
                lum = Math.max(0, Math.min(255, lum));

                const idx = (py * patchSize + px) * 4;
                data[idx] = lum;     // Red
                data[idx + 1] = lum; // Green
                data[idx + 2] = lum; // Blue
                data[idx + 3] = 255; // Alpha
            }
        }

        ctx.putImageData(imgData, 0, 0);
        return offscreen;
    }

    /**
     * Draw stimulus directly onto target canvas at specified visual coordinates
     * @param {CanvasRenderingContext2D} ctx - Target 2D rendering context
     * @param {number} centerX - Canvas center X in pixels
     * @param {number} centerY - Canvas center Y in pixels
     * @param {number} eccDeg - Retinal eccentricity in visual degrees
     * @param {number} angleRad - Polar angle (e.g. 0 = right horizontal, PI = left horizontal)
     * @param {HTMLCanvasElement} dogCanvas - Pre-rendered DoG patch
     */
    drawAtEccentricity(ctx, centerX, centerY, eccDeg, angleRad, dogCanvas) {
        const eccPixels = this.degToPixels(eccDeg);
        const targetX = centerX + eccPixels * Math.cos(angleRad);
        const targetY = centerY + eccPixels * Math.sin(angleRad);

        const patchRadius = Math.floor(dogCanvas.width / 2);
        ctx.drawImage(dogCanvas, targetX - patchRadius, targetY - patchRadius);
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DogRenderer };
}
