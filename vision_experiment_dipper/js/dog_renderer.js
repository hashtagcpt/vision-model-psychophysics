/**
 * dog_renderer.js - Difference of Gaussians (DoG) Stimulus Renderer for Dipper Psychophysics
 * 
 * Supports:
 * - Physiologically balanced ON-center and OFF-center Difference of Gaussians
 * - Pedestal contrast (C_ped) + Increment contrast (Delta_C) composition
 * - Sub-LSB Triangular Probability Density Function (TPDF) spatial dithering
 * - Invariant spatial bounding box clipping
 * - Strict non-overlapping fixation guides (Center 0 deg & Periphery 4 deg)
 */

class DogRenderer {
    /**
     * @param {Object} options
     * @param {number} [options.viewingDistanceCm=57.3] - 57.3 cm ensures 1 cm ~ 1 deg visual angle
     * @param {number} [options.pixelsPerCm=38.0] - Display PPI conversion (~96 DPI default)
     * @param {number} [options.bgLuminance=128] - Neutral mid-gray background (0-255)
     * @param {boolean} [options.enableDithering=true] - Spatial dithering for sub-8-bit contrast
     */
    constructor(options = {}) {
        this.viewingDistanceCm = options.viewingDistanceCm || 57.3;
        this.pixelsPerCm = options.pixelsPerCm || 38.0;
        this.bgLuminance = options.bgLuminance !== undefined ? options.bgLuminance : 128;
        this.enableDithering = options.enableDithering !== undefined ? options.enableDithering : true;

        const rad1Deg = (1.0 * Math.PI) / 180.0;
        const cmPerDeg = 2.0 * this.viewingDistanceCm * Math.tan(rad1Deg / 2.0);
        this.pixelsPerDegree = cmPerDeg * this.pixelsPerCm;
    }

    setPixelsPerDegree(pixelsPerDeg) {
        this.pixelsPerDegree = pixelsPerDeg;
    }

    degToPixels(deg) {
        return deg * this.pixelsPerDegree;
    }

    /**
     * Generate an offscreen canvas containing the DoG patch
     * @param {Object} params
     * @param {number} params.sigmaCenterDeg - Center standard deviation in visual degrees (e.g. 0.25, 0.3)
     * @param {number} params.contrast - Total Michelson contrast (0.0 to 1.0)
     * @param {string} params.polarity - "ON" (bright center) or "OFF" (dark center)
     * @param {number} [params.boxSizePx] - If provided, size of canvas
     * @returns {HTMLCanvasElement}
     */
    createDogCanvas(params) {
        const { sigmaCenterDeg, contrast, polarity } = params;
        const sigmaC = this.degToPixels(sigmaCenterDeg);
        const sigmaS = 2.0 * sigmaC; // Balanced surround ratio

        // Envelope size: 3.5 * sigma_surround covers >= 99.9% of receptive field energy
        const patchRadius = Math.ceil(3.5 * sigmaS);
        const naturalSize = 2 * patchRadius + 1;
        const patchSize = params.boxSizePx ? Math.max(naturalSize, params.boxSizePx) : naturalSize;
        const halfSize = Math.floor(patchSize / 2);

        const offscreen = document.createElement("canvas");
        offscreen.width = patchSize;
        offscreen.height = patchSize;
        const ctx = offscreen.getContext("2d");

        const imgData = ctx.createImageData(patchSize, patchSize);
        const data = imgData.data;

        // Zero DC balance ratio: (sigmaC / sigmaS)^2
        const balanceRatio = Math.pow(sigmaC / sigmaS, 2.0);
        const sign = polarity === "ON" ? 1.0 : -1.0;

        // Peak normalization so maximum absolute amplitude at r=0 equals contrast
        const peakValAtZero = 1.0 - balanceRatio;
        const normFactor = peakValAtZero > 0 ? (1.0 / peakValAtZero) : 1.0;

        const twoSigmaC2 = 2.0 * sigmaC * sigmaC;
        const twoSigmaS2 = 2.0 * sigmaS * sigmaS;

        for (let py = 0; py < patchSize; py++) {
            const dy = py - halfSize;
            const dy2 = dy * dy;
            for (let px = 0; px < patchSize; px++) {
                const dx = px - halfSize;
                const r2 = dx * dx + dy2;

                const gCenter = Math.exp(-r2 / twoSigmaC2);
                const gSurround = Math.exp(-r2 / twoSigmaS2);
                const rawDog = gCenter - balanceRatio * gSurround;
                const normalizedDog = rawDog * normFactor;

                // Contrast delta between -1.0 and +1.0
                const contrastDelta = sign * contrast * normalizedDog;

                // Base continuous luminance
                let lumContinuous = this.bgLuminance * (1.0 + contrastDelta);

                // High-Frequency TPDF Spatial Dithering for sub-LSB contrast resolution
                let lumInt;
                if (this.enableDithering && contrast < 0.20) {
                    const r1 = Math.random();
                    const r2 = Math.random();
                    const tpdfNoise = (r1 + r2 - 1.0) * 0.707;
                    lumInt = Math.round(lumContinuous + tpdfNoise);
                } else {
                    lumInt = Math.round(lumContinuous);
                }

                lumInt = Math.max(0, Math.min(255, lumInt));

                const idx = (py * patchSize + px) * 4;
                data[idx]     = lumInt;
                data[idx + 1] = lumInt;
                data[idx + 2] = lumInt;
                data[idx + 3] = 255;
            }
        }

        ctx.putImageData(imgData, 0, 0);
        return offscreen;
    }

    /**
     * Compute constant bounding box dimension in pixels for a given sigmaCenterDeg
     * @param {number} sigmaCenterDeg 
     * @returns {number}
     */
    getBoundingBoxSizePx(sigmaCenterDeg) {
        const sigmaC = this.degToPixels(sigmaCenterDeg);
        const sigmaS = 2.0 * sigmaC;
        const naturalRadius = Math.ceil(3.5 * sigmaS);
        return (naturalRadius * 2) + 20; // 20px padding
    }

    /**
     * Draw stimulus display with bounding box and non-overlapping fixation guides
     * @param {CanvasRenderingContext2D} ctx - Target canvas context
     * @param {number} screenWidth - Viewport width
     * @param {number} screenHeight - Viewport height
     * @param {Object} options
     * @param {HTMLCanvasElement|null} [options.dogCanvas=null] - Pre-rendered DoG canvas or null for blank
     * @param {number} [options.eccentricityDeg=0] - 0 for Center, 4 for Periphery
     * @param {number} [options.boxSizePx=160] - Bounding box dimension
     * @param {boolean} [options.isIntervalActive=true] - Whether interval is currently active
     */
    drawStimulusDisplay(ctx, screenWidth, screenHeight, options = {}) {
        const {
            dogCanvas = null,
            eccentricityDeg = 0,
            boxSizePx = 160,
            isIntervalActive = true,
            feedback = null // "CORRECT", "INCORRECT", or null
        } = options;

        const centerX = Math.floor(screenWidth / 2);
        const centerY = Math.floor(screenHeight / 2);

        // Fill background with neutral mid-grey
        ctx.fillStyle = `rgb(${this.bgLuminance}, ${this.bgLuminance}, ${this.bgLuminance})`;
        ctx.fillRect(0, 0, screenWidth, screenHeight);

        // Stimulus location:
        // Center: (centerX, centerY)
        // Periphery (4 deg): offset horizontally to the right (+4 deg)
        const eccPx = this.degToPixels(eccentricityDeg);
        const stimX = Math.round(centerX + eccPx);
        const stimY = centerY;

        const halfBox = Math.floor(boxSizePx / 2);
        const boxLeft = stimX - halfBox;
        const boxTop = stimY - halfBox;

        // Feedback color scheme
        let boxStroke = "rgba(45, 45, 45, 0.85)";
        let guideColor = "#1a1a1a";
        let fixCoreColor = "#ffffff";
        if (feedback === "CORRECT") {
            boxStroke = "rgba(46, 125, 50, 0.95)"; // Clear green
            guideColor = "#2e7d32";
            fixCoreColor = "#a5d6a7";
        } else if (feedback === "INCORRECT") {
            boxStroke = "rgba(198, 40, 40, 0.95)"; // Clear red
            guideColor = "#c62828";
            fixCoreColor = "#ef9a9a";
        }

        // 1. Draw Central Fixation Cross
        // If stimulus is in periphery (4 deg), draw persistent central fixation cross at screen center
        if (eccentricityDeg > 0) {
            this._drawCentralFixationCross(ctx, centerX, centerY, guideColor, fixCoreColor);
        }

        // 2. Draw Spatial Bounding Box at stimulus location
        ctx.save();
        ctx.strokeStyle = boxStroke;
        ctx.lineWidth = feedback ? 2.5 : 2.0;
        ctx.strokeRect(boxLeft, boxTop, boxSizePx, boxSizePx);

        // 3. Clip and Draw DoG stimulus inside bounding box
        if (dogCanvas && isIntervalActive) {
            ctx.beginPath();
            ctx.rect(boxLeft + 1, boxTop + 1, boxSizePx - 2, boxSizePx - 2);
            ctx.clip();

            const halfDogW = Math.floor(dogCanvas.width / 2);
            const halfDogH = Math.floor(dogCanvas.height / 2);
            ctx.drawImage(dogCanvas, stimX - halfDogW, stimY - halfDogH);
        }
        ctx.restore();

        // 4. Draw Non-Overlapping Fixation / Alignment Crosshairs
        // Crosshair arms strictly stop outside the bounding box with a 6px clearance gap
        if (eccentricityDeg === 0) {
            // For Center stimulus: 4 outer crosshair arms pointing toward center but stopping outside box
            this._drawOuterCrosshairs(ctx, stimX, stimY, halfBox, guideColor);
        } else {
            // For Peripheral stimulus: subtle outer corner markers pointing to peripheral box
            this._drawPeripheralLocationGuides(ctx, stimX, stimY, halfBox, guideColor);
        }
    }

    /**
     * Draw small central fixation cross at screen center (used during peripheral trials)
     */
    _drawCentralFixationCross(ctx, x, y, strokeColor = "#111111", coreColor = "#ffffff") {
        ctx.save();
        ctx.strokeStyle = strokeColor; // High-visibility fixation mark
        ctx.lineWidth = 2.0;
        const armLength = 8;

        ctx.beginPath();
        ctx.moveTo(x - armLength, y);
        ctx.lineTo(x + armLength, y);
        ctx.moveTo(x, y - armLength);
        ctx.lineTo(x, y + armLength);
        ctx.stroke();

        // Inner core for high contrast on mid-grey
        ctx.strokeStyle = coreColor;
        ctx.lineWidth = 1.0;
        ctx.beginPath();
        ctx.moveTo(x - armLength + 1, y);
        ctx.lineTo(x + armLength - 1, y);
        ctx.moveTo(x, y - armLength + 1);
        ctx.lineTo(x, y + armLength - 1);
        ctx.stroke();
        ctx.restore();
    }

    /**
     * Draw 4 outer crosshair arms pointing toward center, stopping 6px outside the bounding box
     */
    _drawOuterCrosshairs(ctx, centerX, centerY, halfBox, strokeColor = "#1a1a1a") {
        ctx.save();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 2.5;

        const gap = 6;         // 6px clearance gap outside bounding box
        const armLength = 18;   // 18px crosshair arm

        const topInnerY = centerY - halfBox - gap;
        const topOuterY = topInnerY - armLength;

        const bottomInnerY = centerY + halfBox + gap;
        const bottomOuterY = bottomInnerY + armLength;

        const leftInnerX = centerX - halfBox - gap;
        const leftOuterX = leftInnerX - armLength;

        const rightInnerX = centerX + halfBox + gap;
        const rightOuterX = rightInnerX + armLength;

        ctx.beginPath();
        // Top arm
        ctx.moveTo(centerX, topOuterY);
        ctx.lineTo(centerX, topInnerY);
        // Bottom arm
        ctx.moveTo(centerX, bottomInnerY);
        ctx.lineTo(centerX, bottomOuterY);
        // Left arm
        ctx.moveTo(leftOuterX, centerY);
        ctx.lineTo(leftInnerX, centerY);
        // Right arm
        ctx.moveTo(rightInnerX, centerY);
        ctx.lineTo(rightOuterX, centerY);
        ctx.stroke();
        ctx.restore();
    }

    /**
     * Draw subtle alignment corner ticks outside the peripheral bounding box
     */
    _drawPeripheralLocationGuides(ctx, centerX, centerY, halfBox, strokeColor = "rgba(70, 70, 70, 0.6)") {
        ctx.save();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 2.0;

        const gap = 4;
        const tickLen = 8;
        const x1 = centerX - halfBox - gap;
        const x2 = centerX + halfBox + gap;
        const y1 = centerY - halfBox - gap;
        const y2 = centerY + halfBox + gap;

        ctx.beginPath();
        // Top-left corner tick
        ctx.moveTo(x1, y1 + tickLen);
        ctx.lineTo(x1, y1);
        ctx.lineTo(x1 + tickLen, y1);

        // Top-right corner tick
        ctx.moveTo(x2 - tickLen, y1);
        ctx.lineTo(x2, y1);
        ctx.lineTo(x2, y1 + tickLen);

        // Bottom-left corner tick
        ctx.moveTo(x1, y2 - tickLen);
        ctx.lineTo(x1, y2);
        ctx.lineTo(x1 + tickLen, y2);

        // Bottom-right corner tick
        ctx.moveTo(x2 - tickLen, y2);
        ctx.lineTo(x2, y2);
        ctx.lineTo(x2, y2 - tickLen);

        ctx.stroke();
        ctx.restore();
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DogRenderer };
}
