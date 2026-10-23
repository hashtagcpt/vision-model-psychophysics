/**
 * csf_exporter.js - Complete Contrast Sensitivity Function (CSF) CSV/JSON Exporter & Curve Fitter
 * 
 * 1. Exports comprehensive .csv containing full hardware specs, summary CSF table, and raw trial-by-trial logs.
 * 2. Implements parabolic log-sensitivity CSF model fitting (Barten / Movshon standard model):
 *    log10(S(f)) = log10(S_max) - k * (log10(f) - log10(f_max))^2
 * 3. Supports automatic download on experiment completion and manual export.
 */

class CSFExporter {
    constructor() {
        this.trials = [];
        this.metadata = {
            participantId: "SUBJ_001",
            date: new Date().toISOString(),
            viewingDistanceCm: 57.3,
            pixelsPerDegree: 38.0,
            bitDepth: 8,
            ditheringEnabled: true,
            sigmaDeg: 0.5,
            paradigm: "Two-Interval Forced Choice (2IFC 2AFC) Detection with Dual Auditory Cues",
            spatialFrequencies: [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            hardwareProfile: null
        };
    }

    setMetadata(meta) {
        this.metadata = Object.assign(this.metadata, meta);
    }

    setHardwareProfile(profile) {
        this.metadata.hardwareProfile = profile;
    }

    logTrial(trialData) {
        this.trials.push({
            trialIndex: this.trials.length + 1,
            timestamp: new Date().toISOString(),
            ...trialData
        });
    }

    getAccuracyStats() {
        if (this.trials.length === 0) return { totalTrials: 0, nCorrect: 0, accuracyPct: 0 };
        const nCorrect = this.trials.filter(t => t.isCorrect).length;
        return {
            totalTrials: this.trials.length,
            nCorrect: nCorrect,
            accuracyPct: (nCorrect / this.trials.length) * 100
        };
    }

    /**
     * Fit empirical CSF data with the standard parabolic log-sensitivity model:
     * y = a * x^2 + b * x + c
     * where x = log10(f), y = log10(Sensitivity)
     */
    fitCSFModel(csfData) {
        const validPoints = csfData.filter(d => d.sensitivity > 0 && d.spatialFreq > 0);
        const N = validPoints.length;

        if (N < 3) {
            return {
                fitted: false,
                reason: "Insufficient data points for quadratic fit",
                predict: (f) => 10
            };
        }

        let sumX = 0, sumX2 = 0, sumX3 = 0, sumX4 = 0;
        let sumY = 0, sumXY = 0, sumX2Y = 0;

        for (let i = 0; i < N; i++) {
            const x = Math.log10(validPoints[i].spatialFreq);
            const y = Math.log10(validPoints[i].sensitivity);
            const x2 = x * x;
            const x3 = x2 * x;
            const x4 = x2 * x2;

            sumX += x;
            sumX2 += x2;
            sumX3 += x3;
            sumX4 += x4;
            sumY += y;
            sumXY += x * y;
            sumX2Y += x2 * y;
        }

        const M = [
            [sumX4, sumX3, sumX2],
            [sumX3, sumX2, sumX],
            [sumX2, sumX,  N]
        ];

        const det3 = (m) => (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
            m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
            m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        );

        const D = det3(M);

        if (Math.abs(D) < 1e-10) {
            return {
                fitted: false,
                reason: "Singular matrix in polynomial fit",
                predict: (f) => 10
            };
        }

        const Da = det3([
            [sumX2Y, sumX3, sumX2],
            [sumXY,  sumX2, sumX],
            [sumY,   sumX,  N]
        ]);

        const Db = det3([
            [sumX4, sumX2Y, sumX2],
            [sumX3, sumXY,  sumX],
            [sumX2, sumY,   N]
        ]);

        const Dc = det3([
            [sumX4, sumX3, sumX2Y],
            [sumX3, sumX2, sumXY],
            [sumX2, sumX,  sumY]
        ]);

        const a = Da / D;
        const b = Db / D;
        const c = Dc / D;

        let logFMax, fMax, logSMax, sMax, k;
        if (a < 0) {
            logFMax = -b / (2.0 * a);
            fMax = Math.pow(10, logFMax);
            logSMax = c - (b * b) / (4.0 * a);
            sMax = Math.pow(10, logSMax);
            k = -a;
        } else {
            let maxSens = -1;
            let maxF = 1;
            for (const p of validPoints) {
                if (p.sensitivity > maxSens) {
                    maxSens = p.sensitivity;
                    maxF = p.spatialFreq;
                }
            }
            fMax = maxF;
            logFMax = Math.log10(fMax);
            sMax = maxSens;
            logSMax = Math.log10(sMax);
            k = 0.5;
        }

        let ssTot = 0, ssRes = 0;
        const meanY = sumY / N;
        for (let i = 0; i < N; i++) {
            const x = Math.log10(validPoints[i].spatialFreq);
            const y = Math.log10(validPoints[i].sensitivity);
            const yPred = a * x * x + b * x + c;
            ssTot += (y - meanY) * (y - meanY);
            ssRes += (y - yPred) * (y - yPred);
        }
        const rSquared = ssTot > 0 ? Math.max(0, 1.0 - (ssRes / ssTot)) : 1.0;

        return {
            fitted: true,
            coefficients: { a, b, c },
            fMax: fMax,
            sMax: sMax,
            logFMax: logFMax,
            logSMax: logSMax,
            bandwidthK: k,
            rSquared: rSquared,
            predict: (f) => {
                const x = Math.log10(f);
                const logSens = a * x * x + b * x + c;
                return Math.pow(10, logSens);
            }
        };
    }

    /**
     * Generate complete, consolidated .CSV with metadata headers, hardware profile, summary table, and trial records
     * @param {Array<Object>} summaryData
     * @param {Object} modelFit
     * @returns {string} CSV text
     */
    generateFullCSF_CSV(summaryData, modelFit) {
        const acc = this.getAccuracyStats();
        const hw = this.metadata.hardwareProfile || {};
        const d = hw.display || {};
        const g = hw.gpu || {};
        const s = hw.system || {};

        const lines = [];
        lines.push("# ==========================================================================");
        lines.push("# CONTRAST SENSITIVITY FUNCTION (CSF) EXPERIMENT DATA FILE");
        lines.push("# ==========================================================================");
        lines.push(`# Subject Code: ${this.metadata.participantId}`);
        lines.push(`# Session Date: ${this.metadata.date}`);
        lines.push(`# Task Paradigm: ${this.metadata.paradigm}`);
        lines.push("# --------------------------------------------------------------------------");
        lines.push("# HARDWARE & DISPLAY DIAGNOSTIC PROFILE");
        lines.push("# --------------------------------------------------------------------------");
        lines.push(`# Display Resolution: ${d.physicalResolution || 'Unknown'} (CSS: ${d.cssResolution || 'Unknown'}, DPR: ${d.dpr || 1})`);
        lines.push(`# Measured Display Refresh Rate: ${d.refreshRateHz || 'Unknown'} Hz (${d.measuredFps || 'Unknown'} FPS)`);
        lines.push(`# Display Color Depth: ${d.colorDepth || 24}-bit`);
        lines.push(`# Display HDR (High Dynamic Range): ${d.isHDR ? 'ACTIVE / SUPPORTED' : 'INACTIVE / SDR'}`);
        lines.push(`# Detected Color Gamut: ${d.colorGamut || 'sRGB'}`);
        lines.push(`# GPU Renderer: ${g.renderer || 'Unknown'}`);
        lines.push(`# GPU Vendor: ${g.vendor || 'Unknown'}`);
        lines.push(`# WebGL2 10-Bit Integer Buffer (RGB10_A2): ${g.webgl10BitBuffer ? 'SUPPORTED' : 'FALLBACK'}`);
        lines.push(`# WebGL2 Half-Float Buffer (RGBA16F): ${g.webglFloatBuffer ? 'SUPPORTED' : 'FALLBACK'}`);
        lines.push(`# WebGPU Graphics API: ${g.webgpuSupported ? 'SUPPORTED' : 'NOT SUPPORTED'}`);
        lines.push(`# CPU Logical Cores: ${s.cpuCores || 'Unknown'}`);
        lines.push(`# Device Memory (RAM): ${s.deviceMemoryGB ? s.deviceMemoryGB + ' GB' : 'Unknown'}`);
        lines.push(`# Max Touch Points: ${s.maxTouchPoints || 0}`);
        lines.push(`# Platform Architecture: ${s.architecture || 'Unknown'}`);
        lines.push(`# Battery Status: ${s.battery ? s.battery.levelPct + '% (' + (s.battery.charging ? 'Charging' : 'Battery') + ')' : 'AC / N/A'}`);
        lines.push(`# User Agent: "${s.userAgent || 'Unknown'}"`);
        lines.push("# --------------------------------------------------------------------------");
        lines.push("# PSYCHOPHYSICS EXPERIMENT CONFIGURATION");
        lines.push("# --------------------------------------------------------------------------");
        lines.push(`# Viewing Distance: ${this.metadata.viewingDistanceCm} cm`);
        lines.push(`# Calibrated Pixels Per Degree: ${this.metadata.pixelsPerDegree.toFixed(2)} ppd`);
        lines.push(`# Constant Envelope Size: sigma = ${this.metadata.sigmaDeg.toFixed(2)} deg`);
        lines.push(`# Carrier Phase: Randomized trial-to-trial (uniform distribution [0, 360) deg)`);
        lines.push(`# Spatial Presentation: Centered inside spatial bounding box with non-overlapping outer fixation crosshairs`);
        lines.push(`# Fixation Isolation: Fixation cross never overlaps the stimulus grating`);
        lines.push(`# Framebuffer Bit-Depth Mode: ${this.metadata.bitDepth}-bit`);
        lines.push(`# Spatial Noise Dithering: ${this.metadata.ditheringEnabled ? 'ENABLED (Sub-LSB interpolation)' : 'DISABLED'}`);
        lines.push(`# Physical 8-Bit Un-dithered Floor: Contrast = 0.0078125, Sensitivity = 128.0`);
        lines.push(`# Tested Spatial Frequencies (cpd): [${this.metadata.spatialFrequencies.join(', ')}]`);
        lines.push("# --------------------------------------------------------------------------");
        lines.push("# OVERALL RESULTS & MODEL FIT");
        lines.push("# --------------------------------------------------------------------------");
        lines.push(`# Total Trials Completed: ${acc.totalTrials}`);
        lines.push(`# Overall 2AFC Accuracy: ${acc.accuracyPct.toFixed(1)}% (${acc.nCorrect}/${acc.totalTrials})`);
        if (modelFit && modelFit.fitted) {
            lines.push(`# Model Fit Equation: log10(S) = log10(S_max) - k*(log10(f) - log10(f_max))^2`);
            lines.push(`# Peak Contrast Sensitivity (S_max): ${modelFit.sMax.toFixed(2)} (Threshold Contrast = ${(100/modelFit.sMax).toFixed(3)}%)`);
            lines.push(`# Peak Spatial Frequency (f_max): ${modelFit.fMax.toFixed(2)} cpd`);
            lines.push(`# Curve Bandwidth Factor (k): ${modelFit.bandwidthK.toFixed(4)}`);
            lines.push(`# Goodness of Fit (R^2): ${(modelFit.rSquared * 100).toFixed(1)}%`);
        }
        lines.push("# ==========================================================================");
        lines.push("");
        lines.push("# SECTION 1: EMPIRICAL CONTRAST SENSITIVITY FUNCTION SUMMARY TABLE");
        lines.push("spatial_frequency_cpd,sigma_deg,threshold_log10,threshold_std,contrast_threshold_michelson,contrast_sensitivity,ci95_lower_sensitivity,ci95_upper_sensitivity,trials_tested,below_8bit_floor");

        const floor8bit = 1.0 / 128.0;
        summaryData.forEach(d => {
            const belowFloor = d.contrastThreshold < floor8bit ? "YES (Sub-LSB / Dithered)" : "NO";
            const ciLow = 1.0 / Math.pow(10, d.thresholdLog10 + 1.96 * d.thresholdStd);
            const ciHigh = 1.0 / Math.pow(10, d.thresholdLog10 - 1.96 * d.thresholdStd);
            lines.push([
                d.spatialFreqCpd.toFixed(3),
                d.sigmaDeg.toFixed(4),
                d.thresholdLog10.toFixed(4),
                d.thresholdStd.toFixed(4),
                d.contrastThreshold.toFixed(6),
                d.sensitivity.toFixed(2),
                ciLow.toFixed(2),
                ciHigh.toFixed(2),
                d.trialsCount,
                `"${belowFloor}"`
            ].join(","));
        });

        lines.push("");
        lines.push("# ==========================================================================");
        lines.push("# SECTION 2: TRIAL-BY-TRIAL PSYCHOPHYSICAL RAW DATA");
        lines.push("trial_index,timestamp,spatial_frequency_cpd,sigma_deg,target_interval,response_interval,is_correct,reaction_time_ms,test_log10_contrast,test_contrast_michelson,carrier_phase_deg,carrier_phase_rad,estimated_threshold_mean,estimated_threshold_std,estimated_contrast_threshold,estimated_sensitivity,entropy_bits");

        this.trials.forEach(t => {
            const pDeg = t.phaseDeg !== undefined ? t.phaseDeg : (t.phaseRad !== undefined ? t.phaseRad * 180 / Math.PI : 0);
            const pRad = t.phaseRad !== undefined ? t.phaseRad : 0;
            lines.push([
                t.trialIndex,
                `"${t.timestamp}"`,
                t.spatialFreqCpd.toFixed(3),
                t.sigmaDeg.toFixed(4),
                t.targetInterval,
                t.responseInterval,
                t.isCorrect ? 1 : 0,
                t.reactionTimeMs.toFixed(1),
                t.testLog10Contrast.toFixed(4),
                t.testContrast.toFixed(6),
                pDeg.toFixed(1),
                pRad.toFixed(4),
                t.estimatedThresholdMean.toFixed(4),
                t.estimatedThresholdStd.toFixed(4),
                t.estimatedContrastThreshold.toFixed(6),
                t.estimatedSensitivity.toFixed(2),
                t.entropy.toFixed(4)
            ].join(","));
        });

        return lines.join("\n");
    }

    generateTrialCSV() {
        const lines = [];
        lines.push(`# Subject Code: ${this.metadata.participantId}`);
        lines.push(`# Session Date: ${this.metadata.date}`);
        lines.push("trial_index,timestamp,spatial_frequency_cpd,sigma_deg,target_interval,response_interval,is_correct,reaction_time_ms,test_log10_contrast,test_contrast_michelson,carrier_phase_deg,carrier_phase_rad,estimated_threshold_mean,estimated_threshold_std,estimated_contrast_threshold,estimated_sensitivity,entropy_bits");
        this.trials.forEach(t => {
            const pDeg = t.phaseDeg !== undefined ? t.phaseDeg : (t.phaseRad !== undefined ? t.phaseRad * 180 / Math.PI : 0);
            const pRad = t.phaseRad !== undefined ? t.phaseRad : 0;
            lines.push([
                t.trialIndex,
                `"${t.timestamp}"`,
                t.spatialFreqCpd.toFixed(3),
                t.sigmaDeg.toFixed(4),
                t.targetInterval,
                t.responseInterval,
                t.isCorrect ? 1 : 0,
                t.reactionTimeMs.toFixed(1),
                t.testLog10Contrast.toFixed(4),
                t.testContrast.toFixed(6),
                pDeg.toFixed(1),
                pRad.toFixed(4),
                t.estimatedThresholdMean.toFixed(4),
                t.estimatedThresholdStd.toFixed(4),
                t.estimatedContrastThreshold.toFixed(6),
                t.estimatedSensitivity.toFixed(2),
                t.entropy.toFixed(4)
            ].join(","));
        });
        return lines.join("\n");
    }

    generateSummaryCSV(summaryData, modelFit) {
        const lines = [];
        lines.push(`# Subject Code: ${this.metadata.participantId}`);
        lines.push(`# Session Date: ${this.metadata.date}`);
        if (modelFit && modelFit.fitted) {
            lines.push(`# Peak Sensitivity (S_max): ${modelFit.sMax.toFixed(2)} at ${modelFit.fMax.toFixed(2)} cpd, R^2 = ${(modelFit.rSquared * 100).toFixed(1)}%`);
        }
        lines.push("spatial_frequency_cpd,sigma_deg,threshold_log10,threshold_std,contrast_threshold_michelson,contrast_sensitivity,ci95_lower_sensitivity,ci95_upper_sensitivity,trials_tested,below_8bit_floor");
        const floor8bit = 1.0 / 128.0;
        summaryData.forEach(d => {
            const belowFloor = d.contrastThreshold < floor8bit ? "YES" : "NO";
            const ciLow = 1.0 / Math.pow(10, d.thresholdLog10 + 1.96 * d.thresholdStd);
            const ciHigh = 1.0 / Math.pow(10, d.thresholdLog10 - 1.96 * d.thresholdStd);
            lines.push([
                d.spatialFreqCpd.toFixed(3),
                d.sigmaDeg.toFixed(4),
                d.thresholdLog10.toFixed(4),
                d.thresholdStd.toFixed(4),
                d.contrastThreshold.toFixed(6),
                d.sensitivity.toFixed(2),
                ciLow.toFixed(2),
                ciHigh.toFixed(2),
                d.trialsCount,
                `"${belowFloor}"`
            ].join(","));
        });
        return lines.join("\n");
    }

    generateJSON(summaryData, modelFit) {
        return JSON.stringify({
            metadata: this.metadata,
            accuracy: this.getAccuracyStats(),
            modelFit: modelFit && modelFit.fitted ? {
                fMax: modelFit.fMax,
                sMax: modelFit.sMax,
                logFMax: modelFit.logFMax,
                logSMax: modelFit.logSMax,
                bandwidthK: modelFit.bandwidthK,
                rSquared: modelFit.rSquared,
                coefficients: modelFit.coefficients
            } : null,
            csfSummary: summaryData,
            trials: this.trials
        }, null, 2);
    }

    downloadBlob(content, filename, mimeType = "text/csv") {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 600);
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CSFExporter };
}
