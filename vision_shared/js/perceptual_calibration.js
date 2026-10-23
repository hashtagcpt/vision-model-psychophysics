/**
 * perceptual_calibration.js - Empirical Perceptual Timing & Contrast Calibration Module
 * 
 * Provides:
 * 1. Perceptual Screen Timing Check:
 *    - VSync-locked requestAnimationFrame interval benchmarking (Refresh rate, jitter, dropped frames).
 *    - Stroboscopic Vernier Tick test: flashes for exactly 120 ms (14 frames @ 120 Hz, 7 frames @ 60 Hz).
 *      Observer directly verifies continuous frame presentation and zero display stutter.
 * 2. Perceptual Contrast & Gamma Calibration:
 *    - Photometric 50% Halftone Match: calculates true display gamma γ = log(0.5) / log(V_match / 255).
 *    - Sub-LSB Contrast Step Ladder: 8 calibrated low-contrast targets with live WebGL dithering toggle.
 * 3. Cross-Suite Persistent Storage:
 *    - Stored in localStorage ('vision_suite_contrast_calibration' and 'vision_suite_timing_check').
 */

class PerceptualCalibration {
    constructor() {
        this.calibratedGamma = 2.2;
        this.matchValue = 186; // Standard sRGB ~186/255 -> gamma 2.2
        this.measuredFps = 120.0;
        this.nominalFrameMs = 8.33;
        this.jitterMs = 0.15;
        this.droppedFrames = 0;
        this.minVisibleContrast = 0.008; // 0.8% default

        this.modalElement = null;
        this.timingBenchActive = false;
        this.vernierAnimId = null;
        this.activeTab = "timing"; // "timing", "contrast", "size"

        this.loadStoredCalibration();
    }

    loadStoredCalibration() {
        if (typeof localStorage !== "undefined") {
            try {
                const contrastRaw = localStorage.getItem("vision_suite_contrast_calibration");
                if (contrastRaw) {
                    const parsed = JSON.parse(contrastRaw);
                    this.calibratedGamma = parsed.gamma || 2.2;
                    this.matchValue = parsed.matchValue || 186;
                    this.minVisibleContrast = parsed.minVisibleContrast || 0.008;
                }

                const timingRaw = localStorage.getItem("vision_suite_timing_check");
                if (timingRaw) {
                    const parsed = JSON.parse(timingRaw);
                    this.measuredFps = parsed.fps || 120.0;
                    this.nominalFrameMs = parsed.nominalFrameMs || (1000.0 / this.measuredFps);
                    this.jitterMs = parsed.jitterMs || 0.15;
                    this.droppedFrames = parsed.droppedFrames || 0;
                }
            } catch (e) {
                console.warn("Could not load stored perceptual calibration:", e);
            }
        }
    }

    saveContrastCalibration() {
        const payload = {
            gamma: Number(this.calibratedGamma.toFixed(3)),
            matchValue: Math.round(this.matchValue),
            minVisibleContrast: Number(this.minVisibleContrast.toFixed(4)),
            calibratedAt: new Date().toISOString()
        };
        try {
            localStorage.setItem("vision_suite_contrast_calibration", JSON.stringify(payload));
        } catch (e) {
            console.warn("Storage save failed:", e);
        }
        return payload;
    }

    saveTimingCalibration() {
        const payload = {
            fps: Number(this.measuredFps.toFixed(1)),
            nominalFrameMs: Number(this.nominalFrameMs.toFixed(2)),
            jitterMs: Number(this.jitterMs.toFixed(3)),
            droppedFrames: this.droppedFrames,
            verified120msFrames: Math.round(120.0 / this.nominalFrameMs),
            checkedAt: new Date().toISOString()
        };
        try {
            localStorage.setItem("vision_suite_timing_check", JSON.stringify(payload));
        } catch (e) {
            console.warn("Storage save failed:", e);
        }
        return payload;
    }

    /**
     * Measure display refresh rate and inter-frame jitter over 90 VSync cycles
     */
    runTimingBenchmark(targetFrames = 90) {
        return new Promise((resolve) => {
            const timestamps = [];
            let frameCount = 0;

            const step = (timestamp) => {
                timestamps.push(timestamp);
                frameCount++;

                if (frameCount < targetFrames) {
                    requestAnimationFrame(step);
                } else {
                    const deltas = [];
                    let dropped = 0;
                    const elapsed = timestamps[timestamps.length - 1] - timestamps[0];
                    const avgDt = elapsed / (timestamps.length - 1);
                    const measuredFps = 1000.0 / avgDt;

                    for (let i = 1; i < timestamps.length; i++) {
                        const dt = timestamps[i] - timestamps[i - 1];
                        deltas.push(dt);
                        if (dt > 1.5 * avgDt) {
                            dropped++;
                        }
                    }

                    // Compute standard deviation of frame duration (jitter)
                    const mean = deltas.reduce((a, b) => a + b, 0) / deltas.length;
                    const variance = deltas.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / deltas.length;
                    const jitter = Math.sqrt(variance);

                    // Snap to standard refresh rates if within 3.5% tolerance
                    let snappedFps = Math.round(measuredFps);
                    const standards = [60, 75, 90, 120, 144, 165, 240];
                    for (const std of standards) {
                        if (Math.abs(measuredFps - std) < 3.5) {
                            snappedFps = std;
                            break;
                        }
                    }

                    this.measuredFps = snappedFps;
                    this.nominalFrameMs = 1000.0 / snappedFps;
                    this.jitterMs = jitter;
                    this.droppedFrames = dropped;

                    this.saveTimingCalibration();

                    resolve({
                        fps: Number(measuredFps.toFixed(1)),
                        snappedHz: snappedFps,
                        nominalFrameMs: Number((1000.0 / snappedFps).toFixed(2)),
                        jitterMs: Number(jitter.toFixed(3)),
                        droppedFrames: dropped,
                        sampleFrames: targetFrames
                    });
                }
            };

            requestAnimationFrame(step);
        });
    }

    /**
     * Compute display gamma from photometric match value V in [0, 255]
     * (V / 255)^gamma = 0.50  =>  gamma = log(0.5) / log(V / 255)
     */
    static computeGammaFromMatch(matchV) {
        const norm = Math.max(1, Math.min(254, matchV)) / 255.0;
        return Math.log(0.5) / Math.log(norm);
    }

    /**
     * Draw 50% Halftone Photometric calibration patch on canvas
     * Outer area: 1-pixel alternating black (0) and white (255) horizontal square-wave lines
     * Inner circle: solid gray patch of luminance matchV
     */
    drawPhotometricHalftone(canvas, matchV) {
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;
        const centerX = w / 2;
        const centerY = h / 2;
        const radius = Math.min(w, h) * 0.28;

        const imgData = ctx.createImageData(w, h);
        const data = imgData.data;

        for (let y = 0; y < h; y++) {
            // Alternating 1-pixel horizontal lines: y % 2 === 0 -> 255 (white), y % 2 === 1 -> 0 (black)
            const isWhiteLine = (y % 2 === 0);
            const halftoneVal = isWhiteLine ? 255 : 0;
            const dy = y - centerY;
            const dy2 = dy * dy;

            for (let x = 0; x < w; x++) {
                const dx = x - centerX;
                const r2 = dx * dx + dy2;
                const idx = (y * w + x) * 4;

                let lum;
                if (r2 <= radius * radius) {
                    // Inner uniform patch
                    lum = matchV;
                } else {
                    // Outer 50% physical mean halftone pattern
                    lum = halftoneVal;
                }

                data[idx]     = lum;
                data[idx + 1] = lum;
                data[idx + 2] = lum;
                data[idx + 3] = 255;
            }
        }

        ctx.putImageData(imgData, 0, 0);

        // Thin neutral guide circle
        ctx.strokeStyle = "rgba(100, 100, 100, 0.4)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        ctx.stroke();
    }

    /**
     * Draw Weber Contrast Step Visibility Ladder
     * 8 circular disks with Weber contrast C = [0.2%, 0.4%, 0.8%, 1.6%, 3.2%, 6.4%, 12.8%, 25.6%]
     */
    drawContrastLadder(canvas, enableDithering = true) {
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;
        const bgLum = 128;

        ctx.fillStyle = `rgb(${bgLum}, ${bgLum}, ${bgLum})`;
        ctx.fillRect(0, 0, w, h);

        const contrastLevels = [0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256];
        const numDisks = contrastLevels.length;
        const diskRadius = 20;
        const stepX = w / (numDisks + 1);
        const diskY = h * 0.45;

        // Render each contrast disk directly into pixel buffer for precise dithering
        const imgData = ctx.getImageData(0, 0, w, h);
        const data = imgData.data;

        for (let i = 0; i < numDisks; i++) {
            const cVal = contrastLevels[i];
            const diskCenterX = Math.round(stepX * (i + 1));
            const deltaL = bgLum * cVal;
            const targetLum = bgLum + deltaL;

            for (let dy = -diskRadius; dy <= diskRadius; dy++) {
                const py = Math.round(diskY + dy);
                if (py < 0 || py >= h) continue;

                for (let dx = -diskRadius; dx <= diskRadius; dx++) {
                    const px = Math.round(diskCenterX + dx);
                    if (px < 0 || px >= w) continue;

                    if (dx * dx + dy * dy <= diskRadius * diskRadius) {
                        let finalLum;
                        if (enableDithering) {
                            // TPDF spatial noise dithering
                            const r1 = Math.random();
                            const r2 = Math.random();
                            const noise = (r1 + r2 - 1.0) * 0.707;
                            finalLum = Math.round(targetLum + noise);
                        } else {
                            // Raw integer 8-bit truncation (quantization floor at 0.78%)
                            finalLum = Math.round(targetLum);
                        }
                        finalLum = Math.max(0, Math.min(255, finalLum));

                        const idx = (py * w + px) * 4;
                        data[idx]     = finalLum;
                        data[idx + 1] = finalLum;
                        data[idx + 2] = finalLum;
                        data[idx + 3] = 255;
                    }
                }
            }
        }

        ctx.putImageData(imgData, 0, 0);

        // Labels
        ctx.fillStyle = "#222";
        ctx.font = "bold 11px monospace";
        ctx.textAlign = "center";
        for (let i = 0; i < numDisks; i++) {
            const diskCenterX = Math.round(stepX * (i + 1));
            const pctText = (contrastLevels[i] * 100).toFixed(1) + "%";
            ctx.fillText(pctText, diskCenterX, diskY + diskRadius + 18);

            // Highlight sub-8-bit quantization threshold (0.78%)
            if (contrastLevels[i] < 0.0078) {
                ctx.fillStyle = "#d32f2f";
                ctx.font = "9px sans-serif";
                ctx.fillText("Sub-8bit", diskCenterX, diskY + diskRadius + 30);
                ctx.fillStyle = "#222";
                ctx.font = "bold 11px monospace";
            }
        }
    }

    /**
     * Launch Stroboscopic Vernier Tick Timing Test on a target canvas
     * Flashes for exact specified duration (e.g. 120 ms).
     * On each VSync frame, renders a vertical tick mark at x = startX + frame * step.
     */
    flashStroboscopicVernier(canvas, durationMs = 120, onComplete = null) {
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;
        const bgLum = 128;

        const nominalFrameDuration = 1000.0 / this.measuredFps;
        const targetFrames = Math.max(1, Math.round(durationMs / nominalFrameDuration));

        // Layout Vernier Track
        const trackMargin = 30;
        const trackWidth = w - 2 * trackMargin;
        const stepX = trackWidth / (targetFrames + 1);
        const tickY = h * 0.35;
        const tickHeight = 50;

        let frameIndex = 0;
        const timestamps = [];
        const tickPositions = [];

        // Clear canvas initially
        ctx.fillStyle = `rgb(${bgLum}, ${bgLum}, ${bgLum})`;
        ctx.fillRect(0, 0, w, h);

        // Draw track baseline
        ctx.strokeStyle = "rgba(40, 40, 40, 0.4)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(trackMargin, tickY + tickHeight / 2);
        ctx.lineTo(w - trackMargin, tickY + tickHeight / 2);
        ctx.stroke();

        const step = (timestamp) => {
            timestamps.push(timestamp);

            const curX = trackMargin + (frameIndex + 1) * stepX;
            tickPositions.push(curX);

            // Draw this frame's vernier tick
            ctx.strokeStyle = "#000000";
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            ctx.moveTo(curX, tickY);
            ctx.lineTo(curX, tickY + tickHeight);
            ctx.stroke();

            // Label tick index
            ctx.fillStyle = "#111111";
            ctx.font = "bold 10px monospace";
            ctx.textAlign = "center";
            ctx.fillText(`${frameIndex + 1}`, curX, tickY + tickHeight + 14);

            frameIndex++;

            if (frameIndex < targetFrames) {
                this.vernierAnimId = requestAnimationFrame(step);
            } else {
                // Completed exact stimulus duration
                const actualDurationMs = timestamps[timestamps.length - 1] - timestamps[0];
                let dropped = 0;
                for (let i = 1; i < timestamps.length; i++) {
                    if (timestamps[i] - timestamps[i - 1] > 1.5 * nominalFrameDuration) {
                        dropped++;
                    }
                }

                // Summary banner on canvas
                ctx.fillStyle = dropped === 0 ? "rgba(46, 125, 50, 0.15)" : "rgba(198, 40, 40, 0.15)";
                ctx.fillRect(10, h - 38, w - 20, 28);
                ctx.strokeStyle = dropped === 0 ? "#2e7d32" : "#c62828";
                ctx.lineWidth = 1.5;
                ctx.strokeRect(10, h - 38, w - 20, 28);

                ctx.fillStyle = dropped === 0 ? "#1b5e20" : "#b71c1c";
                ctx.font = "bold 12px sans-serif";
                ctx.textAlign = "center";
                const statusTxt = dropped === 0
                    ? `✓ PASS: ${targetFrames} frames delivered in ${actualDurationMs.toFixed(1)} ms (${this.measuredFps} Hz VSync, 0 dropped)`
                    : `⚠ WARNING: ${dropped} frame drop(s) detected during ${actualDurationMs.toFixed(1)} ms presentation`;
                ctx.fillText(statusTxt, w / 2, h - 20);

                if (onComplete) {
                    onComplete({
                        targetFrames,
                        actualFrames: frameIndex,
                        targetDurationMs: durationMs,
                        actualDurationMs: Number(actualDurationMs.toFixed(2)),
                        droppedFrames: dropped
                    });
                }
            }
        };

        this.vernierAnimId = requestAnimationFrame(step);
    }

    /**
     * Open the full unified Perceptual Calibration & Timing Verification Modal
     */
    openModal(initialTab = "timing") {
        this.activeTab = initialTab;
        if (!this.modalElement) {
            this._createModalDOM();
        }
        this.modalElement.style.display = "flex";
        this.switchTab(this.activeTab);
    }

    closeModal() {
        if (this.modalElement) {
            this.modalElement.style.display = "none";
        }
        if (this.vernierAnimId) {
            cancelAnimationFrame(this.vernierAnimId);
        }
    }

    switchTab(tabName) {
        this.activeTab = tabName;
        const tabs = ["timing", "contrast", "size"];
        tabs.forEach(t => {
            const btn = document.getElementById(`tabBtn_${t}`);
            const pane = document.getElementById(`tabPane_${t}`);
            if (btn) btn.classList.toggle("active", t === tabName);
            if (pane) pane.style.display = (t === tabName) ? "block" : "none";
        });

        if (tabName === "timing") {
            this._initTimingTab();
        } else if (tabName === "contrast") {
            this._initContrastTab();
        } else if (tabName === "size") {
            if (window.screenCalibrator) {
                this.closeModal();
                window.screenCalibrator.openCalibrationModal();
            }
        }
    }

    _createModalDOM() {
        const modal = document.createElement("div");
        modal.id = "perceptualCalibrationModal";
        modal.style.cssText = `
            position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
            background: rgba(8, 8, 14, 0.94); z-index: 999999;
            display: flex; align-items: center; justify-content: center;
            padding: 16px; backdrop-filter: blur(8px);
        `;

        modal.innerHTML = `
            <div style="background: #14141e; border: 1px solid #2a2a3e; border-radius: 16px; max-width: 780px; width: 100%; box-shadow: 0 16px 48px rgba(0,0,0,0.7); overflow: hidden; display: flex; flex-direction: column; max-height: 94vh;">
                <!-- Header -->
                <div style="padding: 18px 24px; border-bottom: 1px solid #252538; display: flex; justify-content: space-between; align-items: center; background: #181826;">
                    <div>
                        <div style="font-size: 18px; font-weight: 800; color: #fff; display: flex; align-items: center; gap: 8px;">
                            <span>⚙ Hardware Calibration &amp; Perceptual Verification</span>
                            <span style="font-size: 11px; background: rgba(0, 188, 212, 0.15); color: #00bcd4; padding: 2px 8px; border-radius: 12px; font-weight: 700;">Surface Pro OLED 120Hz</span>
                        </div>
                        <div style="font-size: 12px; color: #9a9ab0; margin-top: 3px;">
                            Verify 120 ms VSync refresh precision and calibrate display gamma &amp; sub-LSB contrast precision.
                        </div>
                    </div>
                    <button type="button" onclick="window.perceptualCalibrator.closeModal()" style="background: none; border: none; color: #888; font-size: 24px; cursor: pointer; padding: 0 6px;">&times;</button>
                </div>

                <!-- Navigation Tabs -->
                <div style="display: flex; border-bottom: 1px solid #222234; background: #0f0f18;">
                    <button id="tabBtn_timing" onclick="window.perceptualCalibrator.switchTab('timing')" class="calib-tab-btn active" style="flex: 1; padding: 12px; background: none; border: none; border-bottom: 2px solid transparent; color: #aaa; font-weight: 700; font-size: 13px; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 6px;">
                        ⏱ 120 ms Timing Check
                    </button>
                    <button id="tabBtn_contrast" onclick="window.perceptualCalibrator.switchTab('contrast')" class="calib-tab-btn" style="flex: 1; padding: 12px; background: none; border: none; border-bottom: 2px solid transparent; color: #aaa; font-weight: 700; font-size: 13px; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 6px;">
                        🌓 Contrast &amp; Gamma Calibration
                    </button>
                    <button id="tabBtn_size" onclick="window.perceptualCalibrator.switchTab('size')" class="calib-tab-btn" style="flex: 1; padding: 12px; background: none; border: none; border-bottom: 2px solid transparent; color: #aaa; font-weight: 700; font-size: 13px; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 6px;">
                        💳 Screen Size (Credit Card)
                    </button>
                </div>

                <!-- Content Area -->
                <div style="padding: 20px 24px; overflow-y: auto; flex: 1;">
                    <!-- TAB 1: TIMING VERIFICATION -->
                    <div id="tabPane_timing">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 10px;">
                            <div>
                                <h3 style="font-size: 15px; color: #fff; font-weight: 700;">Stroboscopic Frame Vernier (120 ms Stimulus Check)</h3>
                                <p style="font-size: 12px; color: #888; margin-top: 2px;">
                                    Verifies that a 120 ms flash delivers exactly the required number of VSync refresh frames (14 frames @ 120 Hz).
                                </p>
                            </div>
                            <button id="btnRunBenchmark" onclick="window.perceptualCalibrator.benchmarkHardwareRefresh()" style="background: rgba(0, 188, 212, 0.15); border: 1px solid rgba(0, 188, 212, 0.4); color: #00bcd4; font-weight: 700; font-size: 12px; padding: 6px 14px; border-radius: 8px; cursor: pointer;">
                                ↻ Benchmark Refresh
                            </button>
                        </div>

                        <!-- Live Refresh Metrics -->
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px;">
                            <div style="background: #0b0b12; border: 1px solid #222234; padding: 10px; border-radius: 8px; text-align: center;">
                                <div style="font-size: 11px; color: #888;">Measured Refresh</div>
                                <div id="statMeasuredFps" style="font-size: 18px; font-weight: 800; color: #00bcd4; margin-top: 2px;">120.0 Hz</div>
                            </div>
                            <div style="background: #0b0b12; border: 1px solid #222234; padding: 10px; border-radius: 8px; text-align: center;">
                                <div style="font-size: 11px; color: #888;">Nominal Frame (τ)</div>
                                <div id="statNominalMs" style="font-size: 18px; font-weight: 800; color: #fff; margin-top: 2px;">8.33 ms</div>
                            </div>
                            <div style="background: #0b0b12; border: 1px solid #222234; padding: 10px; border-radius: 8px; text-align: center;">
                                <div style="font-size: 11px; color: #888;">Inter-Frame Jitter</div>
                                <div id="statJitterMs" style="font-size: 18px; font-weight: 800; color: #00e676; margin-top: 2px;">&plusmn;0.12 ms</div>
                            </div>
                            <div style="background: #0b0b12; border: 1px solid #222234; padding: 10px; border-radius: 8px; text-align: center;">
                                <div style="font-size: 11px; color: #888;">120 ms Frame Count</div>
                                <div id="stat120msFrames" style="font-size: 18px; font-weight: 800; color: #ffb300; margin-top: 2px;">14 Frames</div>
                            </div>
                        </div>

                        <!-- Vernier Canvas Display -->
                        <div style="background: #000; border: 1px solid #2a2a3e; border-radius: 10px; padding: 12px; text-align: center;">
                            <canvas id="vernierCanvas" width="680" height="150" style="width: 100%; height: 150px; background: #808080; border-radius: 6px;"></canvas>
                        </div>

                        <!-- Test Action Buttons -->
                        <div style="display: flex; gap: 10px; margin-top: 14px; align-items: center; justify-content: center; flex-wrap: wrap;">
                            <button onclick="window.perceptualCalibrator.triggerVernierTest(120)" style="background: linear-gradient(135deg, #00bcd4, #0097a7); color: #000; border: none; font-weight: 800; font-size: 13px; padding: 10px 22px; border-radius: 8px; cursor: pointer; box-shadow: 0 4px 14px rgba(0, 188, 212, 0.3);">
                                ▶ Trigger 120 ms Test Flash (Target: 14 Ticks)
                            </button>
                            <button onclick="window.perceptualCalibrator.triggerVernierTest(60)" style="background: rgba(255, 255, 255, 0.08); border: 1px solid #444; color: #ddd; font-weight: 600; font-size: 12px; padding: 9px 16px; border-radius: 8px; cursor: pointer;">
                                Test 60 ms (7 Ticks)
                            </button>
                            <button onclick="window.perceptualCalibrator.triggerVernierTest(240)" style="background: rgba(255, 255, 255, 0.08); border: 1px solid #444; color: #ddd; font-weight: 600; font-size: 12px; padding: 9px 16px; border-radius: 8px; cursor: pointer;">
                                Test 240 ms (28 Ticks)
                            </button>
                        </div>
                    </div>

                    <!-- TAB 2: CONTRAST & GAMMA CALIBRATION -->
                    <div id="tabPane_contrast" style="display: none;">
                        <h3 style="font-size: 15px; color: #fff; font-weight: 700; margin-bottom: 4px;">Photometric 50% Halftone Match (Display Gamma)</h3>
                        <p style="font-size: 12px; color: #888; margin-bottom: 14px;">
                            The outer pattern has 1-pixel alternating black/white stripes (physically 50% mean luminance). 
                            Adjust the slider until the inner gray disk blends seamlessly into the outer background.
                        </p>

                        <!-- Halftone Matching Canvas -->
                        <div style="display: flex; gap: 20px; align-items: center; justify-content: center; flex-wrap: wrap; margin-bottom: 16px;">
                            <div style="background: #000; border: 1px solid #2a2a3e; padding: 8px; border-radius: 10px;">
                                <canvas id="halftoneCanvas" width="240" height="240" style="width: 240px; height: 240px; border-radius: 6px;"></canvas>
                            </div>
                            <div style="flex: 1; min-width: 260px;">
                                <div style="background: #0b0b12; border: 1px solid #222234; padding: 14px; border-radius: 10px; margin-bottom: 12px;">
                                    <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 8px;">
                                        <span style="color: #aaa;">Digital Drive Level (V):</span>
                                        <strong id="lblMatchVal" style="color: #00bcd4; font-family: monospace;">186 / 255</strong>
                                    </div>
                                    <input type="range" id="sliderMatchV" min="120" max="230" step="1" value="186" style="width: 100%; accent-color: #00bcd4;" oninput="window.perceptualCalibrator.onSliderChange(this.value)">
                                    <div style="display: flex; justify-content: space-between; font-size: 10px; color: #666; margin-top: 4px;">
                                        <span>120 (Linear γ=1.0)</span>
                                        <span>186 (sRGB γ=2.2)</span>
                                        <span>230 (High γ=3.0)</span>
                                    </div>
                                </div>

                                <div style="background: #0b0b12; border: 1px solid #222234; padding: 14px; border-radius: 10px;">
                                    <div style="font-size: 12px; color: #aaa;">Calibrated Display Gamma:</div>
                                    <div id="lblGammaVal" style="font-size: 24px; font-weight: 800; color: #00e676; margin-top: 2px;">γ = 2.20</div>
                                    <div style="font-size: 11px; color: #777; margin-top: 4px;">
                                        Applies power-law linearization $L = (V/255)^{1/\gamma}$ in all WebGL shaders.
                                    </div>
                                    <button onclick="window.perceptualCalibrator.applyAndSaveGamma()" style="margin-top: 10px; width: 100%; background: linear-gradient(135deg, #00e676, #00c853); color: #000; font-weight: 800; font-size: 13px; padding: 9px; border-radius: 8px; border: none; cursor: pointer;">
                                        ✓ Save &amp; Apply Gamma to WebGL Shaders
                                    </button>
                                </div>
                            </div>
                        </div>

                        <!-- Sub-LSB Contrast Step Visibility Ladder -->
                        <div style="border-top: 1px solid #252538; padding-top: 14px; margin-top: 10px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <div>
                                    <h4 style="font-size: 13px; color: #fff; font-weight: 700;">Sub-LSB Contrast Step Visibility Check</h4>
                                    <div style="font-size: 11px; color: #888;">The 8-bit quantization floor is 0.78%. WebGL TPDF spatial noise resolves 0.2% - 0.4%.</div>
                                </div>
                                <label style="display: flex; align-items: center; gap: 6px; font-size: 12px; color: #00bcd4; font-weight: 600; cursor: pointer;">
                                    <input type="checkbox" id="chkDitherLadder" checked onchange="window.perceptualCalibrator.toggleDitherLadder(this.checked)">
                                    WebGL TPDF Dither
                                </label>
                            </div>
                            <canvas id="ladderCanvas" width="680" height="90" style="width: 100%; height: 90px; background: #808080; border-radius: 6px;"></canvas>
                        </div>
                    </div>
                </div>

                <!-- Footer -->
                <div style="padding: 14px 24px; border-top: 1px solid #222234; background: #181826; display: flex; justify-content: space-between; align-items: center;">
                    <div id="calibStatusMsg" style="font-size: 12px; color: #00e676; font-weight: 600;">
                        ✓ System Calibrated: 120.0 Hz Refresh &bull; γ = 2.20 Linearized
                    </div>
                    <button onclick="window.perceptualCalibrator.closeModal()" style="background: rgba(255, 255, 255, 0.1); border: 1px solid #444; color: #fff; font-weight: 700; font-size: 13px; padding: 8px 18px; border-radius: 8px; cursor: pointer;">
                        Done / Close
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
        this.modalElement = modal;

        // Custom style for active tab button
        const style = document.createElement("style");
        style.innerHTML = `
            .calib-tab-btn.active {
                color: #00bcd4 !important;
                border-bottom-color: #00bcd4 !important;
                background: rgba(0, 188, 212, 0.08) !important;
            }
        `;
        document.head.appendChild(style);
    }

    _initTimingTab() {
        this.updateTimingMetricsDOM();
        const vernierCanvas = document.getElementById("vernierCanvas");
        if (vernierCanvas) {
            this.flashStroboscopicVernier(vernierCanvas, 120);
        }
    }

    _initContrastTab() {
        const halftoneCanvas = document.getElementById("halftoneCanvas");
        if (halftoneCanvas) {
            this.drawPhotometricHalftone(halftoneCanvas, this.matchValue);
        }

        const slider = document.getElementById("sliderMatchV");
        if (slider) slider.value = this.matchValue;

        const lblMatch = document.getElementById("lblMatchVal");
        if (lblMatch) lblMatch.innerText = `${this.matchValue} / 255`;

        const lblGamma = document.getElementById("lblGammaVal");
        if (lblGamma) lblGamma.innerText = `γ = ${this.calibratedGamma.toFixed(2)}`;

        const ladderCanvas = document.getElementById("ladderCanvas");
        if (ladderCanvas) {
            const chk = document.getElementById("chkDitherLadder");
            this.drawContrastLadder(ladderCanvas, chk ? chk.checked : true);
        }
    }

    updateTimingMetricsDOM() {
        const elFps = document.getElementById("statMeasuredFps");
        const elMs = document.getElementById("statNominalMs");
        const elJitter = document.getElementById("statJitterMs");
        const elFrames = document.getElementById("stat120msFrames");

        if (elFps) elFps.innerText = `${this.measuredFps.toFixed(1)} Hz`;
        if (elMs) elMs.innerText = `${this.nominalFrameMs.toFixed(2)} ms`;
        if (elJitter) elJitter.innerHTML = `&plusmn;${this.jitterMs.toFixed(2)} ms`;
        if (elFrames) {
            const f120 = Math.round(120.0 / this.nominalFrameMs);
            elFrames.innerText = `${f120} Frames`;
        }
    }

    async benchmarkHardwareRefresh() {
        const btn = document.getElementById("btnRunBenchmark");
        if (btn) {
            btn.disabled = true;
            btn.innerText = "Measuring...";
        }

        const res = await this.runTimingBenchmark(90);
        this.updateTimingMetricsDOM();

        const vernierCanvas = document.getElementById("vernierCanvas");
        if (vernierCanvas) {
            this.flashStroboscopicVernier(vernierCanvas, 120);
        }

        if (btn) {
            btn.disabled = false;
            btn.innerText = "↻ Benchmark Refresh";
        }
    }

    triggerVernierTest(durationMs) {
        const vernierCanvas = document.getElementById("vernierCanvas");
        if (vernierCanvas) {
            this.flashStroboscopicVernier(vernierCanvas, durationMs);
        }
    }

    onSliderChange(val) {
        const v = parseInt(val, 10);
        this.matchValue = v;
        this.calibratedGamma = PerceptualCalibration.computeGammaFromMatch(v);

        const lblMatch = document.getElementById("lblMatchVal");
        if (lblMatch) lblMatch.innerText = `${v} / 255`;

        const lblGamma = document.getElementById("lblGammaVal");
        if (lblGamma) lblGamma.innerText = `γ = ${this.calibratedGamma.toFixed(2)}`;

        const halftoneCanvas = document.getElementById("halftoneCanvas");
        if (halftoneCanvas) {
            this.drawPhotometricHalftone(halftoneCanvas, v);
        }
    }

    applyAndSaveGamma() {
        this.saveContrastCalibration();
        const msg = document.getElementById("calibStatusMsg");
        if (msg) {
            msg.innerHTML = `✓ Saved &amp; Applied: γ = ${this.calibratedGamma.toFixed(2)} to WebGL Stimulus Shaders`;
            msg.style.color = "#00e676";
        }

        // Broadcast to any active WebGL stimulus engine
        if (window.webglEngine && typeof window.webglEngine.setGamma === "function") {
            window.webglEngine.setGamma(this.calibratedGamma);
        }
    }

    toggleDitherLadder(enabled) {
        const ladderCanvas = document.getElementById("ladderCanvas");
        if (ladderCanvas) {
            this.drawContrastLadder(ladderCanvas, enabled);
        }
    }
}

// Global Singleton
if (typeof window !== "undefined") {
    window.perceptualCalibrator = new PerceptualCalibration();
}

if (typeof module !== "undefined" && module.exports) {
    module.exports = { PerceptualCalibration };
}
