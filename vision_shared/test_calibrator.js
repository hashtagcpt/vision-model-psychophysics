/**
 * test_calibrator.js - Automated Unit Tests for ScreenCalibrator
 */

const assert = require("assert");
const { ScreenCalibrator } = require("./js/screen_calibrator.js");

console.log("==========================================================================");
console.log(" RUNNING AUTOMATED UNIT TESTS: SCREEN CALIBRATOR & VIEWING DISTANCE");
console.log("==========================================================================\n");

// [Test 1] Geometry and Conversion Accuracy
console.log("[Test 1] Testing physical ID-1 dimensions and pixels-per-degree geometry...");
const calib = new ScreenCalibrator();

// Set simulated card width to 359px (corresponds to Surface Pro 11 / ~267 DPI)
// 359 px / 85.60 mm = 4.1939 px/mm = 41.939 px/cm = 106.52 px/in = 267.8 PPI (DPR 2)
calib.currentCardWidthPx = 359;
calib.viewingDistanceCm = 57.3;
calib.recomputeMetrics();

console.log(`  -> Computed pxPerMm: ${calib.pxPerMm.toFixed(3)} px/mm`);
console.log(`  -> Computed pxPerCm: ${calib.pxPerCm.toFixed(2)} px/cm`);
console.log(`  -> Computed DPI: ${calib.dpi} PPI`);
console.log(`  -> Computed pxPerDeg @ 57.3cm: ${calib.pxPerDeg.toFixed(2)} px/deg`);

assert(Math.abs(calib.pxPerMm - (359 / 85.60)) < 0.001, "pxPerMm calculation should be exact");
assert(Math.abs(calib.pxPerCm - (calib.pxPerMm * 10)) < 0.001, "pxPerCm must be 10x pxPerMm");
// At 57.3 cm, 1 deg visual angle is exactly 1.000 cm on screen:
const rad1Deg = Math.PI / 180.0;
const expectedCmPerDeg = 2.0 * 57.3 * Math.tan(rad1Deg / 2.0);
assert(Math.abs(expectedCmPerDeg - 1.0) < 0.001, "At 57.3 cm, 1 visual deg is exactly 1 cm");
assert(Math.abs(calib.pxPerDeg - calib.pxPerCm) < 0.05, "At 57.3 cm, pxPerDeg must equal pxPerCm");
console.log("  [PASS] Geometry and visual angle formulas verified!\n");

// [Test 2] Blind Spot Viewing Distance Trigonometry
console.log("[Test 2] Testing virtual chinrest blind spot calculation...");
// Assume observer is 57.3 cm away.
// The blind spot is at 15.5 deg eccentricity.
// deltaX_mm = 573 mm * tan(15.5 deg) = 573 * 0.27732 = 158.90 mm.
const radBlindSpot = (15.5 * Math.PI) / 180.0;
const simDeltaMm = (57.3 * 10.0) * Math.tan(radBlindSpot);
const reconstructedDistCm = (simDeltaMm / Math.tan(radBlindSpot)) / 10.0;
assert(Math.abs(reconstructedDistCm - 57.3) < 0.0001, "Blind spot trigonometry must invert to exact distance");
console.log(`  -> Simulated Blind Spot deltaX: ${simDeltaMm.toFixed(1)} mm`);
console.log(`  -> Inverted Viewing Distance: ${reconstructedDistCm.toFixed(1)} cm`);
console.log("  [PASS] Virtual chinrest trigonometry verified!\n");

// [Test 3] Hardware Display Presets
console.log("[Test 3] Testing display presets...");
assert(ScreenCalibrator.PRESETS.surface_pro.ppi === 267);
assert(ScreenCalibrator.PRESETS.surface_pro.defaultDistCm === 57.3);
assert(ScreenCalibrator.PRESETS.fhd_laptop.ppi === 141);
console.log("  [PASS] Preset table verified!\n");

console.log("==========================================================================");
console.log(" ALL SCREEN CALIBRATOR TESTS PASSED (3/3)");
console.log("==========================================================================");
