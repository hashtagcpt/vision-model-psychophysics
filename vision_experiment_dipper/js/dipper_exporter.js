/**
 * dipper_exporter.js - Data Exporter for Dipper Functions (TvC) Experiment
 * 
 * Generates:
 * 1. Consolidated Full CSV (Hardware metadata + Dipper summary + Raw trial logs)
 * 2. Summary CSV (Empirical Dipper points, 95% CIs, facilitation metrics)
 * 3. Trial-by-Trial CSV (Every individual 2IFC presentation)
 * 4. Full Session JSON (Complete machine-readable dataset)
 */

class DipperExporter {
    constructor() {
        this.trials = [];
        this.hardwareProfile = null;
        this.subjectCode = "SUBJ_UNKNOWN";
        this.sessionStartTime = new Date();
    }

    setSubjectCode(code) {
        this.subjectCode = (code || "SUBJ_UNKNOWN").trim().toUpperCase();
    }

    setHardwareProfile(profile) {
        this.hardwareProfile = profile;
    }

    logTrial(trialData) {
        this.trials.push({
            trialIndex: this.trials.length + 1,
            timestamp: new Date().toISOString(),
            polarity: trialData.polarity,
            eccentricityDeg: trialData.eccentricityDeg,
            pedestalContrast: trialData.pedestalContrast,
            targetInterval: trialData.targetInterval,
            responseInterval: trialData.responseInterval,
            isCorrect: trialData.isCorrect ? 1 : 0,
            reactionTimeMs: Math.round(trialData.reactionTimeMs || 0),
            testIncrementLog10: Number(trialData.testIncrementLog10.toFixed(4)),
            testIncrementContrast: Number(trialData.testIncrementContrast.toFixed(5)),
            totalTargetContrast: Number((trialData.pedestalContrast + trialData.testIncrementContrast).toFixed(5)),
            estimatedThresholdMean: Number(trialData.estimatedThresholdMean.toFixed(4)),
            estimatedThresholdStd: Number(trialData.estimatedThresholdStd.toFixed(4)),
            estimatedIncrementContrast: Number(Math.pow(10, trialData.estimatedThresholdMean).toFixed(5)),
            entropyBits: Number(trialData.entropyBits.toFixed(4))
        });
    }

    generateFullCSV(summaryData, modelFits = {}) {
        const lines = [];

        // --- SECTION 0: METADATA & HARDWARE PROFILE ---
        lines.push("# ==========================================================================");
        lines.push("# VISION SCIENCE EXPERIMENT: DIPPER FUNCTIONS (TvC) CONTRAST DISCRIMINATION");
        lines.push("# Difference of Gaussians (DoG) Stimuli: ON vs. OFF in Center & 4° Periphery");
        lines.push("# Watson (2017) Multidimensional QUEST+ Bayesian Adaptive Threshold Estimation");
        lines.push("# ==========================================================================");
        lines.push(`# Subject Code: ${this.subjectCode}`);
        lines.push(`# Session Date: ${this.sessionStartTime.toISOString()}`);
        lines.push("# Task Paradigm: Two-Interval Forced Choice (2IFC 2AFC) Contrast Increment Detection");
        lines.push("# Chance Level: 50.0% (gamma = 0.50)");
        lines.push("# Stimulus Envelope: Balanced Difference of Gaussians (sigma_s = 2.0 * sigma_c, 0 net DC)");
        lines.push("# Spatial Bounding Box: Invariant neutral dark-grey bounding frame");
        lines.push("# Fixation Isolation: Central & peripheral non-overlapping crosshair guides");
        lines.push("# Sub-LSB Resolution: Triangular Probability Density Function (TPDF) spatial dithering");

        if (this.hardwareProfile) {
            lines.push("# --------------------------------------------------------------------------");
            lines.push("# HARDWARE & DISPLAY DIAGNOSTIC PROFILE");
            lines.push("# --------------------------------------------------------------------------");
            lines.push(`# Display Physical Resolution: ${this.hardwareProfile.screen?.physicalResolution || "Unknown"}`);
            lines.push(`# Device Pixel Ratio (DPR): ${this.hardwareProfile.screen?.devicePixelRatio || "Unknown"}`);
            lines.push(`# Measured Screen Refresh Rate: ${this.hardwareProfile.screen?.measuredFps ? this.hardwareProfile.screen.measuredFps + " Hz" : "Unknown"}`);
            lines.push(`# High Dynamic Range (HDR): ${this.hardwareProfile.screen?.isHighDynamicRange ? "ACTIVE (HDR Supported)" : "Standard Dynamic Range (SDR)"}`);
            lines.push(`# Color Gamut: ${this.hardwareProfile.screen?.colorGamut || "sRGB"}`);
            lines.push(`# Screen Color Depth: ${this.hardwareProfile.screen?.colorDepth ? this.hardwareProfile.screen.colorDepth + "-bit" : "Unknown"}`);
            lines.push(`# GPU Renderer: ${this.hardwareProfile.gpu?.unmaskedRenderer || "Unknown"}`);
            lines.push(`# GPU Vendor: ${this.hardwareProfile.gpu?.unmaskedVendor || "Unknown"}`);
            lines.push(`# WebGL2 RGB10_A2 (10-bit int): ${this.hardwareProfile.gpu?.webgl2_rgb10a2 ? "SUPPORTED" : "Not Supported"}`);
            lines.push(`# WebGL2 RGBA16F (Float): ${this.hardwareProfile.gpu?.webgl2_float ? "SUPPORTED" : "Not Supported"}`);
            lines.push(`# WebGPU Graphics API: ${this.hardwareProfile.gpu?.webgpuAvailable ? "Available" : "Not Supported"}`);
            lines.push(`# CPU Logical Cores: ${this.hardwareProfile.platform?.logicalCores || "Unknown"}`);
            lines.push(`# Device RAM Estimate: ${this.hardwareProfile.platform?.deviceMemoryGB ? this.hardwareProfile.platform.deviceMemoryGB + " GB" : "Unknown"}`);
            lines.push(`# Touch Points: ${this.hardwareProfile.platform?.maxTouchPoints || 0}`);
            lines.push(`# Platform Architecture: ${this.hardwareProfile.platform?.platform || "Unknown"}`);
            lines.push(`# User Agent: ${this.hardwareProfile.platform?.userAgent || "Unknown"}`);
        }

        // --- MODEL FIT METRICS ---
        if (modelFits && Object.keys(modelFits).length > 0) {
            lines.push("# --------------------------------------------------------------------------");
            lines.push("# FOLEY TRANSDUCER DIPPER MODEL FITS");
            lines.push("# --------------------------------------------------------------------------");
            for (const [condName, fit] of Object.entries(modelFits)) {
                lines.push(`# Condition [${condName}]: Delta_C0 = ${fit.deltaC0?.toFixed(4)}, Min_Delta_C = ${fit.minDeltaC?.toFixed(4)}, Facilitation_Ratio = ${fit.facilitationRatio?.toFixed(2)}x, Weber_Slope = ${fit.weberSlope?.toFixed(3)}, R2 = ${(fit.r2 * 100)?.toFixed(1)}%`);
            }
        }
        lines.push("");

        // --- SECTION 1: SUMMARY DIPPER TABLE ---
        lines.push("# --- SECTION 1: EMPIRICAL DIPPER (TvC) SUMMARY TABLE ---");
        const summaryHeaders = [
            "condition_id",
            "polarity",
            "eccentricity_deg",
            "pedestal_contrast",
            "pedestal_log10",
            "increment_threshold_log10",
            "increment_threshold_std",
            "increment_threshold_contrast",
            "ci95_lower_increment",
            "ci95_upper_increment",
            "trials_tested",
            "is_facilitated",
            "facilitation_factor"
        ];
        lines.push(summaryHeaders.join(","));

        for (const row of summaryData) {
            const pedLog = row.pedestalContrast > 0 ? Math.log10(row.pedestalContrast).toFixed(4) : "-inf";
            const rowVals = [
                row.conditionId,
                row.polarity,
                row.eccentricityDeg,
                row.pedestalContrast.toFixed(4),
                pedLog,
                row.thresholdLog10.toFixed(4),
                row.thresholdStd.toFixed(4),
                row.incrementThresholdContrast.toFixed(5),
                row.ci95Lower.toFixed(5),
                row.ci95Upper.toFixed(5),
                row.trialsTested,
                row.isFacilitated ? "YES" : "NO",
                row.facilitationFactor ? row.facilitationFactor.toFixed(2) : "1.00"
            ];
            lines.push(rowVals.join(","));
        }
        lines.push("");

        // --- SECTION 2: RAW TRIAL DATA ---
        lines.push("# --- SECTION 2: TRIAL-BY-TRIAL PSYCHOPHYSICAL RAW DATA ---");
        const trialHeaders = [
            "trial_index",
            "timestamp",
            "polarity",
            "eccentricity_deg",
            "pedestal_contrast",
            "target_interval",
            "response_interval",
            "is_correct",
            "reaction_time_ms",
            "test_increment_log10",
            "test_increment_contrast",
            "total_target_contrast",
            "estimated_threshold_mean",
            "estimated_threshold_std",
            "estimated_increment_contrast",
            "entropy_bits"
        ];
        lines.push(trialHeaders.join(","));

        for (const t of this.trials) {
            const vals = [
                t.trialIndex,
                t.timestamp,
                t.polarity,
                t.eccentricityDeg,
                t.pedestalContrast,
                t.targetInterval,
                t.responseInterval,
                t.isCorrect,
                t.reactionTimeMs,
                t.testIncrementLog10,
                t.testIncrementContrast,
                t.totalTargetContrast,
                t.estimatedThresholdMean,
                t.estimatedThresholdStd,
                t.estimatedIncrementContrast,
                t.entropyBits
            ];
            lines.push(vals.join(","));
        }

        return lines.join("\r\n");
    }

    generateSummaryCSV(summaryData, modelFits = {}) {
        const lines = [];
        lines.push("# DIPPER SUMMARY TABLE: Difference of Gaussians TvC");
        lines.push(`# Subject Code: ${this.subjectCode}`);
        lines.push(`# Date: ${this.sessionStartTime.toISOString()}`);
        lines.push("condition_id,polarity,eccentricity_deg,pedestal_contrast,pedestal_log10,increment_threshold_log10,increment_threshold_std,increment_threshold_contrast,ci95_lower,ci95_upper,trials_tested,is_facilitated,facilitation_factor");
        
        for (const row of summaryData) {
            const pedLog = row.pedestalContrast > 0 ? Math.log10(row.pedestalContrast).toFixed(4) : "-inf";
            lines.push([
                row.conditionId,
                row.polarity,
                row.eccentricityDeg,
                row.pedestalContrast.toFixed(4),
                pedLog,
                row.thresholdLog10.toFixed(4),
                row.thresholdStd.toFixed(4),
                row.incrementThresholdContrast.toFixed(5),
                row.ci95Lower.toFixed(5),
                row.ci95Upper.toFixed(5),
                row.trialsTested,
                row.isFacilitated ? "YES" : "NO",
                row.facilitationFactor ? row.facilitationFactor.toFixed(2) : "1.00"
            ].join(","));
        }
        return lines.join("\r\n");
    }

    generateTrialCSV() {
        const lines = [];
        lines.push("# DIPPER TRIAL-BY-TRIAL RAW DATA");
        lines.push(`# Subject Code: ${this.subjectCode}`);
        lines.push(`# Date: ${this.sessionStartTime.toISOString()}`);
        lines.push("trial_index,timestamp,polarity,eccentricity_deg,pedestal_contrast,target_interval,response_interval,is_correct,reaction_time_ms,test_increment_log10,test_increment_contrast,total_target_contrast,estimated_threshold_mean,estimated_threshold_std,estimated_increment_contrast,entropy_bits");
        for (const t of this.trials) {
            lines.push([
                t.trialIndex,
                t.timestamp,
                t.polarity,
                t.eccentricityDeg,
                t.pedestalContrast,
                t.targetInterval,
                t.responseInterval,
                t.isCorrect,
                t.reactionTimeMs,
                t.testIncrementLog10,
                t.testIncrementContrast,
                t.totalTargetContrast,
                t.estimatedThresholdMean,
                t.estimatedThresholdStd,
                t.estimatedIncrementContrast,
                t.entropyBits
            ].join(","));
        }
        return lines.join("\r\n");
    }

    generateJSON(summaryData, modelFits = {}) {
        return JSON.stringify({
            metadata: {
                experiment: "Dipper Functions (TvC) Contrast Discrimination",
                paradigm: "2IFC 2AFC Difference of Gaussians (ON vs. OFF in Center & 4° Periphery)",
                subjectCode: this.subjectCode,
                sessionStartTime: this.sessionStartTime.toISOString(),
                totalTrials: this.trials.length
            },
            hardwareProfile: this.hardwareProfile,
            modelFits: modelFits,
            summary: summaryData,
            trials: this.trials
        }, null, 2);
    }

    downloadFile(content, filename, mimeType = "text/csv;charset=utf-8;") {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.setAttribute("href", url);
        link.setAttribute("download", filename);
        link.style.visibility = "hidden";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    downloadFullCSV(summaryData, modelFits) {
        const csv = this.generateFullCSV(summaryData, modelFits);
        const filename = `dipper_data_${this.subjectCode}_${this._getDateSlug()}.csv`;
        this.downloadFile(csv, filename, "text/csv;charset=utf-8;");
    }

    downloadSummaryCSV(summaryData, modelFits) {
        const csv = this.generateSummaryCSV(summaryData, modelFits);
        const filename = `dipper_summary_${this.subjectCode}_${this._getDateSlug()}.csv`;
        this.downloadFile(csv, filename, "text/csv;charset=utf-8;");
    }

    downloadTrialCSV() {
        const csv = this.generateTrialCSV();
        const filename = `dipper_trials_${this.subjectCode}_${this._getDateSlug()}.csv`;
        this.downloadFile(csv, filename, "text/csv;charset=utf-8;");
    }

    downloadJSON(summaryData, modelFits) {
        const json = this.generateJSON(summaryData, modelFits);
        const filename = `dipper_session_${this.subjectCode}_${this._getDateSlug()}.json`;
        this.downloadFile(json, filename, "application/json;charset=utf-8;");
    }

    downloadPlotPNG(canvasElement) {
        const link = document.createElement("a");
        link.download = `dipper_plot_${this.subjectCode}_${this._getDateSlug()}.png`;
        link.href = canvasElement.toDataURL("image/png");
        link.click();
    }

    _getDateSlug() {
        const now = new Date();
        const y = now.getFullYear();
        const m = String(now.getMonth() + 1).padStart(2, "0");
        const d = String(now.getDate()).padStart(2, "0");
        const h = String(now.getHours()).padStart(2, "0");
        const min = String(now.getMinutes()).padStart(2, "0");
        return `${y}${m}${d}_${h}${min}`;
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DipperExporter };
}
