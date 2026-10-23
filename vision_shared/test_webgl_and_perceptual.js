/**
 * test_webgl_and_perceptual.js - Unit tests for WebGLStimulusEngine & PerceptualCalibration
 */

const assert = require("assert");
const { WebGLStimulusEngine } = require("./js/webgl_stimulus_engine.js");
const { PerceptualCalibration } = require("./js/perceptual_calibration.js");

console.log("==========================================================================");
console.log(" RUNNING AUTOMATED UNIT TESTS: WEBGL STIMULUS ENGINE & PERCEPTUAL CALIBRATION");
console.log("==========================================================================\n");

// Test 1: VSync Target Frame Timing Math
console.log("[Test 1] Testing VSync frame calculation for exact 120 ms presentations...");
const frames120Hz = WebGLStimulusEngine.calculateTargetFrames(120, 120.0);
const frames60Hz = WebGLStimulusEngine.calculateTargetFrames(120, 60.0);
const frames144Hz = WebGLStimulusEngine.calculateTargetFrames(120, 144.0);

console.log(`  -> 120 ms @ 120 Hz: ${frames120Hz} frames (Expected: 14)`);
console.log(`  -> 120 ms @ 60 Hz:  ${frames60Hz} frames (Expected: 7)`);
console.log(`  -> 120 ms @ 144 Hz: ${frames144Hz} frames (Expected: 17)`);

assert.strictEqual(frames120Hz, 14, "120 ms at 120 Hz must equal 14 VSync frames");
assert.strictEqual(frames60Hz, 7, "120 ms at 60 Hz must equal 7 VSync frames");
assert.strictEqual(frames144Hz, 17, "120 ms at 144 Hz must equal 17 VSync frames");
console.log("  [PASS] VSync frame timing math verified!\n");

// Test 2: Gabor Mathematical Luminance & Envelope Modulation
console.log("[Test 2] Testing Gabor GLSL mathematical ground truth...");
const gaborCenter = WebGLStimulusEngine.computeGaborLuminance(100, 100, {
    centerX: 100,
    centerY: 100,
    spatialFreqPixels: 0.05,
    contrast: 0.50,
    orientationRad: 0,
    phaseRad: 0,
    sigmaPixels: 20,
    bgLuminance: 128
});
// At center (r=0), envelope=1, carrier=cos(0)=1, L = 128 * (1 + 0.50 * 1 * 1) = 192
console.log(`  -> Peak luminance at carrier crest: ${gaborCenter} (Expected: 192)`);
assert.ok(Math.abs(gaborCenter - 192.0) < 0.001, "Peak Gabor luminance should match 192.0");

const gaborTrough = WebGLStimulusEngine.computeGaborLuminance(100, 100, {
    centerX: 100,
    centerY: 100,
    spatialFreqPixels: 0.05,
    contrast: 0.50,
    orientationRad: 0,
    phaseRad: Math.PI, // Half-cycle phase shift
    sigmaPixels: 20,
    bgLuminance: 128
});
console.log(`  -> Trough luminance at carrier trough: ${gaborTrough} (Expected: 64)`);
assert.ok(Math.abs(gaborTrough - 64.0) < 0.001, "Trough Gabor luminance should match 64.0");
console.log("  [PASS] Gabor mathematical formulas verified!\n");

// Test 3: DoG Balanced Contrast & Pedestal Composition
console.log("[Test 3] Testing Difference of Gaussians (DoG) & Pedestal math...");
const dogCenterON = WebGLStimulusEngine.computeDogLuminance(50, 50, {
    centerX: 50,
    centerY: 50,
    sigmaCPixels: 10,
    sigmaSPixels: 20,
    totalContrast: 0.20,
    polarity: "ON",
    bgLuminance: 128
});
// Center: g_c=1, g_s=1, balanceRatio = (10/20)^2 = 0.25, peakNorm = 1 / (1 - 0.25) = 1.3333
// wave = (1 - 0.25) * 1.3333 = 1.0. Delta = 1.0 * 0.20 * 1.0 = 0.20
// L = 128 * (1 + 0.20) = 153.6
console.log(`  -> ON-center peak luminance: ${dogCenterON.toFixed(2)} (Expected: 153.6)`);
assert.ok(Math.abs(dogCenterON - 153.6) < 0.01, "ON-center peak luminance must match 153.6");

const dogCenterOFF = WebGLStimulusEngine.computeDogLuminance(50, 50, {
    centerX: 50,
    centerY: 50,
    sigmaCPixels: 10,
    sigmaSPixels: 20,
    totalContrast: 0.20,
    polarity: "OFF",
    bgLuminance: 128
});
console.log(`  -> OFF-center trough luminance: ${dogCenterOFF.toFixed(2)} (Expected: 102.4)`);
assert.ok(Math.abs(dogCenterOFF - 102.4) < 0.01, "OFF-center trough luminance must match 102.4");
console.log("  [PASS] DoG mathematical formulas verified!\n");

// Test 4: Photometric Halftone Matching & Display Gamma Inversion
console.log("[Test 4] Testing Photometric Halftone match gamma determination...");
// For standard sRGB gamma 2.2: (V / 255)^2.2 = 0.50 => V = 255 * (0.50)^(1/2.2) = 186.13
const gammaComputed = PerceptualCalibration.computeGammaFromMatch(186);
console.log(`  -> Gamma from V=186: ${gammaComputed.toFixed(3)} (Expected ~2.203)`);
assert.ok(Math.abs(gammaComputed - 2.20) < 0.05, "Match value 186 must yield gamma ~2.2");

// Linear display gamma 1.0: V = 255 * 0.5 = 127.5
const gammaLinear = PerceptualCalibration.computeGammaFromMatch(128);
console.log(`  -> Gamma from V=128: ${gammaLinear.toFixed(3)} (Expected ~1.00)`);
assert.ok(Math.abs(gammaLinear - 1.00) < 0.05, "Match value 128 must yield gamma ~1.0");

// Power-law Gamma linearization check
const linLum = WebGLStimulusEngine.applyGamma(128, 2.2);
console.log(`  -> Linearized midpoint pixel value: ${linLum.toFixed(1)}`);
assert.ok(linLum > 0 && linLum < 255, "Linearized luminance must remain bounded");
console.log("  [PASS] Photometric gamma calibration verified!\n");

console.log("==========================================================================");
console.log(" ALL WEBGL & PERCEPTUAL CALIBRATION UNIT TESTS PASSED (4/4)");
console.log("==========================================================================");
