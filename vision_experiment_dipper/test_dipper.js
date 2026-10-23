/**
 * test_dipper.js - Automated Unit & Integration Tests for Dipper Functions Module
 */

const assert = require("assert");
const { QuestPlus } = require("./js/questplus.js");
const { DogRenderer } = require("./js/dog_renderer.js");
const { DipperExporter } = require("./js/dipper_exporter.js");

console.log("==========================================================================");
console.log(" RUNNING AUTOMATED UNIT & INTEGRATION TESTS: DIPPER FUNCTIONS EXPERIMENT");
console.log("==========================================================================\n");

// [Test 1] QuestPlus Bayesian Convergence
console.log("[Test 1] Testing QuestPlus 2AFC initialization & Bayesian update...");
const qp = new QuestPlus({
    thresholds: [-2.5, -2.0, -1.5, -1.0],
    intensities: [-2.5, -2.0, -1.5, -1.0],
    guessRate: 0.5,
    lapseRate: 0.02
});
assert.strictEqual(qp.nT, 4);
const initialMean = qp.getThresholdMean();
assert(Math.abs(initialMean - (-1.75)) < 0.01, "Initial prior mean should be -1.75");

// Provide correct responses at -1.5
qp.update(-1.5, 1);
qp.update(-1.5, 1);
qp.update(-1.5, 1);
const updatedMean = qp.getThresholdMean();
assert(updatedMean < initialMean, "Threshold mean should decrease after consecutive correct responses");
console.log(`  -> QuestPlus updated correctly: Initial = ${initialMean.toFixed(2)}, Updated = ${updatedMean.toFixed(2)} [PASS]`);

// [Test 2] DogRenderer Configuration & Geometry
console.log("\n[Test 2] Testing DogRenderer geometry & bounding box calculation...");
const renderer = new DogRenderer({
    viewingDistanceCm: 57.3,
    pixelsPerCm: 38.0
});
const degPixels = renderer.degToPixels(1.0);
assert(degPixels > 30 && degPixels < 50, `1 deg visual angle should be ~38px, got ${degPixels}`);

const boxSizeCenter = renderer.getBoundingBoxSizePx(0.25);
const boxSizePeriph = renderer.getBoundingBoxSizePx(0.35);
assert(boxSizeCenter >= 40, `Bounding box size should be substantial (got ${boxSizeCenter})`);
assert(boxSizePeriph > boxSizeCenter, "Peripheral larger envelope should produce larger bounding box");
console.log(`  -> DogRenderer geometry verified: Box 0.25° = ${boxSizeCenter}px, Box 0.35° = ${boxSizePeriph}px [PASS]`);

// [Test 3] DipperExporter Full CSV Generation
console.log("\n[Test 3] Testing DipperExporter full CSV output & metadata headers...");
const exporter = new DipperExporter();
exporter.setSubjectCode("SUBJ_20260926_TEST");
exporter.setHardwareProfile({
    screen: {
        physicalResolution: "2880 x 1920",
        devicePixelRatio: 2.0,
        measuredFps: 120.0,
        isHighDynamicRange: true,
        colorGamut: "rec2020",
        colorDepth: 30
    },
    gpu: {
        unmaskedRenderer: "Qualcomm Adreno X1-85 GPU",
        unmaskedVendor: "Qualcomm",
        webgl2_rgb10a2: true,
        webgl2_float: true,
        webgpuAvailable: true
    },
    platform: {
        logicalCores: 8,
        deviceMemoryGB: 16,
        maxTouchPoints: 10,
        platform: "ARM64",
        userAgent: "SurfaceProTest"
    }
});

// Log sample trials
exporter.logTrial({
    polarity: "ON",
    eccentricityDeg: 0,
    pedestalContrast: 0.04,
    targetInterval: 1,
    responseInterval: 1,
    isCorrect: true,
    reactionTimeMs: 412,
    testIncrementLog10: -2.35,
    testIncrementContrast: 0.00447,
    estimatedThresholdMean: -2.35,
    estimatedThresholdStd: 0.12,
    entropyBits: 3.14
});

const sampleSummary = [
    {
        conditionId: "ON_ecc0_ped0",
        polarity: "ON",
        eccentricityDeg: 0,
        pedestalContrast: 0.0,
        pedestalIndex: 0,
        thresholdLog10: -1.82,
        thresholdStd: 0.14,
        incrementThresholdContrast: 0.0151,
        ci95Lower: 0.011,
        ci95Upper: 0.021,
        trialsTested: 15,
        isFacilitated: false,
        facilitationFactor: 1.0
    },
    {
        conditionId: "ON_ecc0_ped3",
        polarity: "ON",
        eccentricityDeg: 0,
        pedestalContrast: 0.04,
        pedestalIndex: 3,
        thresholdLog10: -2.45,
        thresholdStd: 0.11,
        incrementThresholdContrast: 0.00355,
        ci95Lower: 0.0026,
        ci95Upper: 0.0049,
        trialsTested: 15,
        isFacilitated: true,
        facilitationFactor: 4.25
    }
];

const modelFits = {
    "ON (Center 0°)": {
        deltaC0: 0.0151,
        minDeltaC: 0.00355,
        facilitationRatio: 4.25,
        weberSlope: 0.65,
        r2: 0.98
    }
};

const fullCsv = exporter.generateFullCSV(sampleSummary, modelFits);
assert(fullCsv.includes("# Subject Code: SUBJ_20260926_TEST"), "CSV must include subject code");
assert(fullCsv.includes("Qualcomm Adreno X1-85 GPU"), "CSV must include GPU renderer");
assert(fullCsv.includes("SECTION 1: EMPIRICAL DIPPER (TvC) SUMMARY TABLE"), "CSV must contain summary table");
assert(fullCsv.includes("SECTION 2: TRIAL-BY-TRIAL PSYCHOPHYSICAL RAW DATA"), "CSV must contain trial raw data");
assert(fullCsv.includes("ON_ecc0_ped3"), "CSV must contain summary row");
assert(fullCsv.includes("4.25"), "CSV must contain facilitation factor");
console.log("  -> CSV generation verified with complete headers, summary table & raw trial logs [PASS]");

// [Test 4] JSON Data Export
console.log("\n[Test 4] Testing JSON data serialization...");
const jsonStr = exporter.generateJSON(sampleSummary, modelFits);
const parsed = JSON.parse(jsonStr);
assert.strictEqual(parsed.metadata.subjectCode, "SUBJ_20260926_TEST");
assert.strictEqual(parsed.summary.length, 2);
assert.strictEqual(parsed.trials.length, 1);
assert.strictEqual(parsed.trials[0].isCorrect, 1);
console.log("  -> JSON export structure verified [PASS]");

// [Test 5] QuestPlus New Methods (getOptimalStimulus, getEntropy, getCredibleInterval)
console.log("\n[Test 5] Testing QuestPlus methods getOptimalStimulus(), getEntropy(), getCredibleInterval()...");
assert(typeof qp.getOptimalStimulus === "function", "getOptimalStimulus must be a function");
const optStim = qp.getOptimalStimulus();
assert(typeof optStim === "number" && !isNaN(optStim), "Optimal stimulus should be a valid number");

assert(typeof qp.getEntropy === "function", "getEntropy must be a function");
const entropy = qp.getEntropy();
assert(typeof entropy === "number" && entropy >= 0, "Entropy should be non-negative");

assert(typeof qp.getCredibleInterval === "function", "getCredibleInterval must be a function");
const ci = qp.getCredibleInterval(0.95);
assert(ci && typeof ci.lower === "number" && typeof ci.upper === "number", "CI should have numeric lower and upper");
assert(ci.lower <= ci.upper, "CI lower bound must be <= upper bound");
console.log(`  -> New QuestPlus methods verified: optimal = ${optStim.toFixed(2)}, entropy = ${entropy.toFixed(3)} bits, CI = [${ci.lower.toFixed(2)}, ${ci.upper.toFixed(2)}] [PASS]`);

console.log("\n==========================================================================");
console.log(" ALL DIPPER UNIT & INTEGRATION TESTS PASSED (5/5)");
console.log("==========================================================================");
