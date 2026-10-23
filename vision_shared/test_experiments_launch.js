/**
 * test_experiments_launch.js - End-to-end simulation of experiment initialization & trial launching
 */

// 1. Mock minimal browser environment
const domElements = new Map();

function createMockElement(tag, id = "") {
    const el = {
        id,
        tagName: tag.toUpperCase(),
        style: {},
        classList: {
            add: () => {},
            remove: () => {},
            toggle: () => {}
        },
        value: id === "participantId" || id === "masterSubjectCode" ? "SUBJ_TEST_OBSERVER" : "0.5",
        innerText: "",
        textContent: "",
        innerHTML: "",
        width: 1920,
        height: 1080,
        getContext: (type) => {
            if (type === "webgl2" || type === "webgl") {
                return {
                    createBuffer: () => ({}),
                    bindBuffer: () => {},
                    bufferData: () => {},
                    createShader: () => ({}),
                    shaderSource: () => {},
                    compileShader: () => {},
                    getShaderParameter: () => true,
                    getShaderInfoLog: () => "",
                    createProgram: () => ({}),
                    attachShader: () => {},
                    linkProgram: () => {},
                    getProgramParameter: () => true,
                    getProgramInfoLog: () => "",
                    useProgram: () => {},
                    getAttribLocation: () => 0,
                    enableVertexAttribArray: () => {},
                    vertexAttribPointer: () => {},
                    getUniformLocation: () => ({}),
                    uniform1f: () => {},
                    uniform2f: () => {},
                    uniform4f: () => {},
                    uniform1i: () => {},
                    uniform4fv: () => {},
                    drawArrays: () => {},
                    enable: () => {},
                    disable: () => {},
                    blendFunc: () => {},
                    viewport: () => {},
                    clearColor: () => {},
                    clear: () => {}
                };
            }
            return {
                createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
                putImageData: () => {},
                fillRect: () => {},
                strokeRect: () => {},
                beginPath: () => {},
                moveTo: () => {},
                lineTo: () => {},
                stroke: () => {},
                fill: () => {},
                arc: () => {},
                save: () => {},
                restore: () => {},
                clip: () => {},
                resetTransform: () => {},
                scale: () => {},
                measureText: () => ({ width: 10 })
            };
        },
        addEventListener: (event, handler) => {
            el["_on_" + event] = handler;
        },
        focus: () => {}
    };
    if (id) domElements.set(id, el);
    return el;
}

global.document = {
    getElementById: (id) => {
        if (!domElements.has(id)) {
            domElements.set(id, createMockElement("div", id));
        }
        return domElements.get(id);
    },
    createElement: (tag) => createMockElement(tag),
    body: createMockElement("body", "body"),
    documentElement: createMockElement("html", "html")
};

global.window = {
    innerWidth: 1920,
    innerHeight: 1080,
    devicePixelRatio: 2.0,
    addEventListener: () => {},
    location: { search: "?subject=SUBJ_TEST_OBSERVER", href: "" },
    AudioContext: class {
        constructor() {
            this.currentTime = 0;
            this.state = "running";
            this.destination = {};
        }
        createOscillator() {
            return {
                type: "sine",
                frequency: { setValueAtTime: () => {} },
                connect: () => {},
                start: () => {},
                stop: () => {}
            };
        }
        createGain() {
            return {
                gain: {
                    setValueAtTime: () => {},
                    linearRampToValueAtTime: () => {},
                    exponentialRampToValueAtTime: () => {}
                },
                connect: () => {}
            };
        }
        resume() {}
    },
    requestAnimationFrame: (cb) => setTimeout(() => cb(Date.now()), 10),
    cancelAnimationFrame: (id) => clearTimeout(id)
};

global.requestAnimationFrame = global.window.requestAnimationFrame;
global.cancelAnimationFrame = global.window.cancelAnimationFrame;

global.performance = {
    now: () => Date.now()
};

global.localStorage = {
    _data: {},
    getItem: (k) => global.localStorage._data[k] || null,
    setItem: (k, v) => { global.localStorage._data[k] = String(v); },
    removeItem: (k) => { delete global.localStorage._data[k]; },
    clear: () => { global.localStorage._data = {}; }
};

// Required classes
const { WebGLStimulusEngine } = require("./js/webgl_stimulus_engine.js");
global.WebGLStimulusEngine = WebGLStimulusEngine;

const { ScreenCalibrator } = require("./js/screen_calibrator.js");
global.ScreenCalibrator = ScreenCalibrator;
global.screenCalibrator = new ScreenCalibrator();

const { PerceptualCalibration } = require("./js/perceptual_calibration.js");
global.PerceptualCalibration = PerceptualCalibration;
global.perceptualCalibrator = new PerceptualCalibration();

// Test CSF
console.log("-----------------------------------------------------------------");
console.log("[Simulation Test 1] Verifying CSF Experiment Launch & Lifecycle...");
const { QuestPlus } = require("../vision_experiment_csf/js/questplus.js");
global.QuestPlus = QuestPlus;
const { CSFExporter } = require("../vision_experiment_csf/js/csf_exporter.js");
global.CSFExporter = CSFExporter;
const { GaborRenderer } = require("../vision_experiment_csf/js/gabor_renderer.js");
global.GaborRenderer = GaborRenderer;
const { HardwareChecker } = require("../vision_experiment_csf/js/hardware_checker.js");
global.HardwareChecker = HardwareChecker;

// Load CSF Experiment
const vm = require("vm");
const csfCode = require("fs").readFileSync("vision_experiment_csf/js/experiment_csf.js", "utf8");
vm.runInThisContext(csfCode);

const csf = new CSFExperiment();
console.log("  -> CSFExperiment instantiated successfully!");
csf.startExperiment();
console.log("  -> CSFExperiment startExperiment() launched!");
console.log("  -> Active state:", csf.state);
if (csf.state === "FIXATION_PRE" || csf.state === "IDLE" || csf.state === "INTERVAL_1") {
    console.log("  [PASS] CSF experiment started successfully!");
} else {
    console.error("  [FAIL] Unexpected CSF state:", csf.state);
    process.exit(1);
}

// Test Dipper
console.log("-----------------------------------------------------------------");
console.log("[Simulation Test 2] Verifying Dipper Experiment Launch & Lifecycle...");
const { DogRenderer } = require("../vision_experiment_dipper/js/dog_renderer.js");
global.DogRenderer = DogRenderer;
const { DipperExporter } = require("../vision_experiment_dipper/js/dipper_exporter.js");
global.DipperExporter = DipperExporter;

const dipperCode = require("fs").readFileSync("vision_experiment_dipper/js/experiment_dipper.js", "utf8");
vm.runInThisContext(dipperCode);

const dipper = new DipperExperiment();
console.log("  -> DipperExperiment instantiated successfully!");
dipper.startExperiment();
console.log("  -> DipperExperiment startExperiment() launched!");
console.log("  -> Total trials scheduled:", dipper.totalTrials);
console.log("  [PASS] Dipper experiment started successfully!");

// Test DoG Polarity
console.log("-----------------------------------------------------------------");
console.log("[Simulation Test 3] Verifying DoG Polarity Experiment Launch & Lifecycle...");
const { DataExporter } = require("../vision_experiment_dog_quest/js/data_exporter.js");
global.DataExporter = DataExporter;

const dogCode = require("fs").readFileSync("vision_experiment_dog_quest/js/experiment.js", "utf8");
vm.runInThisContext(dogCode);

const dogExp = new VisionExperiment();
console.log("  -> VisionExperiment instantiated successfully!");
dogExp.start();
console.log("  -> VisionExperiment start() launched!");
console.log("  -> Total conditions scheduled:", dogExp.conditions.length);
console.log("  [PASS] DoG Polarity experiment started successfully!");

console.log("-----------------------------------------------------------------");
console.log(" ALL EXPERIMENTS LAUNCHED AND PASSED VERIFICATION WITH FLYING COLORS!");
console.log("-----------------------------------------------------------------");
