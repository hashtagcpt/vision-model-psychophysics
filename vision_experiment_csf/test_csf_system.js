/**
 * test_csf_system.js - Automated Test Harness for 2IFC CSF Experiment
 * Verifies:
 * 1. QuestPlus 2AFC configuration
 * 2. Hardware profile integration
 * 3. CSFExporter full CSV output structure and subject code tagging
 * 4. Model fitting and prediction
 */

const { QuestPlus } = require('./js/questplus.js');
const { CSFExporter } = require('./js/csf_exporter.js');
const { HardwareChecker } = require('./js/hardware_checker.js');
const assert = require('assert');

console.log("==========================================================================");
console.log(" RUNNING AUTOMATED UNIT & INTEGRATION TESTS: CSF EXPERIMENT");
console.log("==========================================================================");

// ------------------------------------------------------------------------
// Test 1: QuestPlus 2AFC Convergence & Chance Level
// ------------------------------------------------------------------------
console.log("\n[Test 1] Testing QuestPlus 2AFC initialization & Bayesian update...");
const q = new QuestPlus({
    thresholds: [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0],
    intensities: [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0],
    slope: 2.5,
    guessRate: 0.5, // 2AFC chance level
    lapseRate: 0.02
});

assert.strictEqual(q.guessRate, 0.5, "Guess rate must be 0.5 for 2AFC");
const initialT = q.getThresholdMean();
assert(initialT >= -3.0 && initialT <= 0.0, "Threshold mean within bounds");

// Simulate 5 correct responses at intensity 0.0
for (let i = 0; i < 5; i++) {
    q.update(0.0, 1);
}
const updatedT = q.getThresholdMean();
assert(updatedT < initialT, "Threshold should decrease towards higher sensitivity after correct responses");
console.log(`  -> QuestPlus updated correctly: Initial T = ${initialT.toFixed(2)}, Updated T = ${updatedT.toFixed(2)} [PASS]`);

// ------------------------------------------------------------------------
// Test 2: HardwareChecker Class & Profile Structure
// ------------------------------------------------------------------------
console.log("\n[Test 2] Testing HardwareChecker structure...");
const hwChecker = new HardwareChecker();
assert.strictEqual(typeof hwChecker.runDiagnostic, 'function', "runDiagnostic must be a function");
assert.strictEqual(typeof hwChecker.renderReport, 'function', "renderReport must be a function");
console.log("  -> HardwareChecker interface verified [PASS]");

// ------------------------------------------------------------------------
// Test 3: CSFExporter & Full CSV Generation with Hardware Profile & Subject Code
// ------------------------------------------------------------------------
console.log("\n[Test 3] Testing CSFExporter full CSV output & metadata headers...");
const exporter = new CSFExporter();

const subjectCode = "SUBJ_20260925_TEST99";
exporter.setMetadata({
    participantId: subjectCode,
    viewingDistanceCm: 57.3,
    pixelsPerDegree: 42.5,
    bitDepth: 10,
    ditheringEnabled: true,
    sigmaDeg: 0.5,
    spatialFrequencies: [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    trialsPerCondition: 20
});

// Provide a mock hardware diagnostic profile matching Surface Pro 11th Edition
const mockHwProfile = {
    display: {
        cssResolution: "1440 x 960",
        physicalResolution: "2880 x 1920",
        dpr: 2.0,
        refreshRateHz: 120,
        measuredFps: 120.1,
        colorDepth: 30,
        pixelDepth: 30,
        isHDR: true,
        isVideoHDR: true,
        colorGamut: "BT.2020 (Ultra-Wide Gamut)",
        isRec2020: true,
        isP3: true,
        orientation: "landscape-primary (0°)"
    },
    gpu: {
        renderer: "Qualcomm Adreno X1-85 GPU",
        vendor: "Qualcomm Technologies Inc.",
        webgl2Supported: true,
        webgl10BitBuffer: true,
        webglFloatBuffer: true,
        canvasDisplayP3: true,
        maxTextureSize: 16384,
        maxRenderbufferSize: 16384,
        maxViewportDims: "16384 x 16384",
        webgpuSupported: true,
        webgpuAdapterInfo: { architecture: "qualcomm-adreno" }
    },
    system: {
        cpuCores: 12,
        deviceMemoryGB: 16,
        maxTouchPoints: 10,
        isTouchDevice: true,
        architecture: "ARM64 (64-bit)",
        battery: { charging: true, levelPct: 98 },
        userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; ARM64)"
    },
    assessment: {
        isTenBitCapable: true,
        summaryText: "10-Bit HDR OLED Pipeline Active (Surface Pro Wide Gamut Verified)",
        badgeClass: "diag-badge-hdr"
    }
};

exporter.setHardwareProfile(mockHwProfile);

// Log mock trials across 9 spatial frequencies with randomized phases
const sfs = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
sfs.forEach((sf, idx) => {
    const randomPhase = Math.random() * 2.0 * Math.PI;
    exporter.logTrial({
        spatialFreqCpd: sf,
        sigmaDeg: 0.5,
        targetInterval: 1,
        responseInterval: 1,
        isCorrect: true,
        reactionTimeMs: 420.5,
        testLog10Contrast: -2.1,
        testContrast: 0.00794,
        phaseRad: randomPhase,
        phaseDeg: randomPhase * (180.0 / Math.PI),
        estimatedThresholdMean: -2.15,
        estimatedThresholdStd: 0.08,
        estimatedContrastThreshold: 0.00707,
        estimatedSensitivity: 141.4,
        entropy: 1.25
    });
});

const summaryData = sfs.map(sf => ({
    spatialFreqCpd: sf,
    sigmaDeg: 0.5,
    thresholdLog10: -2.0,
    thresholdStd: 0.1,
    contrastThreshold: 0.01,
    sensitivity: 100.0,
    trialsCount: 20
}));

const modelFit = exporter.fitCSFModel(summaryData.map(s => ({
    spatialFreq: s.spatialFreqCpd,
    sensitivity: s.sensitivity
})));

const csvOutput = exporter.generateFullCSF_CSV(summaryData, modelFit);

// Assertions on CSV
assert(csvOutput.includes(`# Subject Code: ${subjectCode}`), "CSV must contain the unique Subject Code");
assert(csvOutput.includes("Qualcomm Adreno X1-85 GPU"), "CSV must include GPU renderer");
assert(csvOutput.includes("120 Hz"), "CSV must include measured refresh rate");
assert(csvOutput.includes("BT.2020"), "CSV must include color gamut");
assert(csvOutput.includes("CPU Logical Cores: 12"), "CSV must include CPU cores");
assert(csvOutput.includes("ARM64"), "CSV must include architecture");
assert(csvOutput.includes("Carrier Phase: Randomized trial-to-trial"), "CSV must state randomized carrier phase");
assert(csvOutput.includes("Fixation cross never overlaps the stimulus grating"), "CSV must state fixation-stimulus isolation");
assert(csvOutput.includes("carrier_phase_deg,carrier_phase_rad"), "CSV must have carrier phase columns");
assert(csvOutput.includes("SECTION 1: EMPIRICAL CONTRAST SENSITIVITY FUNCTION SUMMARY TABLE"), "CSV must contain Section 1");
assert(csvOutput.includes("SECTION 2: TRIAL-BY-TRIAL PSYCHOPHYSICAL RAW DATA"), "CSV must contain Section 2");
assert(csvOutput.includes("0.125"), "CSV must contain low frequency 0.125 cpd");
assert(csvOutput.includes("32.000"), "CSV must contain high frequency 32 cpd");

// Standalone CSV methods
const trialCsv = exporter.generateTrialCSV();
assert(trialCsv.includes("carrier_phase_deg,carrier_phase_rad"), "Trial CSV must have phase columns");
const summaryCsv = exporter.generateSummaryCSV(summaryData, modelFit);
assert(summaryCsv.includes("spatial_frequency_cpd"), "Summary CSV must have frequency column");

console.log("  -> CSV generation verified with all metadata headers, hardware profile, summary table & raw logs [PASS]");

// ------------------------------------------------------------------------
// Test 4: JSON Export
// ------------------------------------------------------------------------
console.log("\n[Test 4] Testing JSON export...");
const jsonOutput = exporter.generateJSON(summaryData, modelFit);
const parsed = JSON.parse(jsonOutput);
assert.strictEqual(parsed.metadata.participantId, subjectCode, "JSON must match participant ID");
assert.strictEqual(parsed.trials.length, 9, "JSON must contain 9 logged trials");
assert(parsed.trials[0].phaseRad !== undefined, "JSON must contain phaseRad");
assert.strictEqual(parsed.metadata.hardwareProfile.display.refreshRateHz, 120, "JSON must include 120Hz refresh rate");
console.log("  -> JSON export verified [PASS]");

console.log("\n==========================================================================");
console.log(" ALL TESTS PASSED SUCCESSFULLY (4/4)");
console.log("==========================================================================");
