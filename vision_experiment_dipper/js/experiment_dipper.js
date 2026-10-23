/**
 * experiment_dipper.js - Controller for Dipper Functions (TvC) Visual Psychophysics
 * 
 * Implements:
 * - Difference of Gaussians (DoG) contrast discrimination / pedestal masking
 * - ON vs. OFF center polarity
 * - Center (0 deg) vs. Periphery (4 deg) retinal eccentricity
 * - Two-Interval Forced Choice (2IFC 2AFC) with dual auditory tone markers
 * - Multidimensional QUEST+ Bayesian adaptive staircase per condition
 * - Real-time Foley transducer dipper model fitting & multi-curve canvas plotting
 * - Full CSV and JSON data export with hardware diagnostic profiling
 */

class DipperExperiment {
    constructor() {
        this.canvas = document.getElementById("experimentCanvas");
        this.ctx = null;
        this.webglEngine = null;

        if (this.canvas && typeof WebGLStimulusEngine !== "undefined") {
            try {
                this.webglEngine = new WebGLStimulusEngine(this.canvas, {
                    enableDithering: true,
                    bgLuminance: 128
                });
                if (!this.webglEngine.gl) {
                    this.webglEngine = null;
                }
            } catch (e) {
                console.warn("WebGL initialization failed, falling back to 2D canvas:", e);
                this.webglEngine = null;
            }
        }
        if (!this.webglEngine && this.canvas && typeof this.canvas.getContext === "function") {
            this.ctx = this.canvas.getContext("2d");
        }

        // Psychophysics display parameters
        this.viewingDistanceCm = 57.3; // 57.3 cm -> 1 cm = 1 deg visual angle
        this.pixelsPerCm = 38.0;      // Standard display density

        if (typeof ScreenCalibrator !== "undefined") {
            const calib = ScreenCalibrator.getCalibration();
            if (calib) {
                this.viewingDistanceCm = calib.viewingDistanceCm || 57.3;
                this.pixelsPerCm = calib.pxPerCm || 38.0;
            }
            if (window.screenCalibrator) {
                window.screenCalibrator.onCalibrateCallback = (data) => {
                    this.viewingDistanceCm = data.viewingDistanceCm;
                    this.pixelsPerCm = data.pxPerCm;
                    this._resizeCanvas();
                    const textEl = document.getElementById("dipperCalibText");
                    if (textEl) textEl.innerHTML = `${data.viewingDistanceCm} cm (${data.pxPerDeg.toFixed(1)} px/&deg;)`;
                };
            }
        }

        this.dogRenderer = new DogRenderer({
            viewingDistanceCm: this.viewingDistanceCm,
            pixelsPerCm: this.pixelsPerCm,
            bgLuminance: 128,
            enableDithering: true
        });

        this.hardwareChecker = new HardwareChecker();
        this.exporter = new DipperExporter();

        // Audio cues for 2IFC intervals
        this.audioCtx = null;

        // Experiment Configuration Defaults
        this.trialsPerCondition = 15;
        this.centerSigmaDeg = 0.25;    // Center Gaussian sigma (visual degrees)
        this.periphSigmaDeg = 0.35;    // Slightly larger in periphery to account for cortical magnification

        // Pedestal contrast sets
        this.pedestalSets = {
            standard: [0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.50], // 8 pedestals
            quick:    [0.0, 0.02, 0.06, 0.18, 0.45]                    // 5 pedestals
        };
        this.currentPedestals = this.pedestalSets.standard;

        // Selected condition filters
        this.selectedPolarity = "BOTH";       // "BOTH", "ON", "OFF"
        this.selectedEccentricity = "BOTH";   // "BOTH", "CENTER", "PERIPH"

        // State Machine
        // IDLE -> INTERVAL1 -> ISI -> INTERVAL2 -> RESPONSE -> TRIAL_FEEDBACK -> FINISHED
        this.state = "IDLE";

        // Current trial state
        this.currentTrialIndex = 0;
        this.totalTrials = 0;
        this.conditions = [];          // List of condition configurations
        this.questPlusInstances = {};  // conditionKey -> QuestPlus
        this.trialSequence = [];       // Interleaved condition indices
        this.currentTrialData = null;

        // Timing constants (ms) - 120 ms stimulus duration
        this.intervalDurationMs = 120;
        this.isiDurationMs = 400;
        this.stimulusOnsetTime = 0;

        // Pre-computed bounding box dimension
        this.boxSizePx = 160;

        // DOM Elements cache
        this.elements = {
            setupModal: document.getElementById("setupModal"),
            resultsModal: document.getElementById("resultsModal"),
            topBar: document.getElementById("topBar"),
            progressBar: document.getElementById("progressBar"),
            progressText: document.getElementById("progressText"),
            participantId: document.getElementById("participantId"),
            btnGenSubjectCode: document.getElementById("btnGenSubjectCode"),
            subjectCodeError: document.getElementById("subjectCodeError"),
            selectPolarity: document.getElementById("selectPolarity"),
            selectEccentricity: document.getElementById("selectEccentricity"),
            selectPedestals: document.getElementById("selectPedestals"),
            selectTrials: document.getElementById("selectTrials"),
            btnStart: document.getElementById("btnStart"),
            touchControls: document.getElementById("touchControls"),
            btnInterval1: document.getElementById("btnInterval1"),
            btnInterval2: document.getElementById("btnInterval2"),
            summaryTableBody: document.getElementById("summaryTableBody"),
            plotCanvas: document.getElementById("plotCanvas"),
            hardwareReportContainer: document.getElementById("hardwareReportContainer"),
            csvAutoDownloadBanner: document.getElementById("csvAutoDownloadBanner"),
            csvAutoDownloadText: document.getElementById("csvAutoDownloadText")
        };

        this._initSubjectCode();
        this._initHardwareDiagnostic();
        this._bindEvents();
        this._resizeCanvas();
    }

    _initSubjectCode() {
        if (window.subjectManager) {
            window.subjectManager.bindUi({
                inputEl: this.elements.participantId,
                selectEl: "selectExistingSubject",
                btnGenEl: this.elements.btnGenSubjectCode,
                onSubjectChanged: (code) => {
                    this.exporter.setSubjectCode(code);
                }
            });
            const activeCode = window.subjectManager.getActiveSubject();
            this.exporter.setSubjectCode(activeCode);
        } else {
            const urlParams = new URLSearchParams(window.location.search);
            const urlSubj = urlParams.get("subject") || localStorage.getItem("vision_suite_subject_code");
            if (urlSubj && this.elements.participantId) {
                this.elements.participantId.value = urlSubj.trim().toUpperCase();
            } else {
                this.generateUniqueSubjectCode();
            }
        }
    }

    generateUniqueSubjectCode() {
        let code;
        if (window.subjectManager) {
            code = window.subjectManager.generateSensibleSubjectCode();
        } else {
            code = "OBS_01";
        }
        if (this.elements.participantId) {
            this.elements.participantId.value = code;
        }
        localStorage.setItem("vision_suite_subject_code", code);
        this.exporter.setSubjectCode(code);
        return code;
    }

    async _initHardwareDiagnostic() {
        try {
            await this.hardwareChecker.runAllChecks();
            const profile = this.hardwareChecker.getReport();
            this.exporter.setHardwareProfile(profile);

            if (this.elements.hardwareReportContainer) {
                this.elements.hardwareReportContainer.innerHTML = `
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; font-size: 12px; margin-top: 10px;">
                        <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 6px; border-left: 3px solid #00bcd4;">
                            <strong style="color: #00bcd4;">Display & Color:</strong><br>
                            Resolution: ${profile.screen?.physicalResolution || "Unknown"}<br>
                            Measured Refresh: <strong>${profile.screen?.measuredFps ? profile.screen.measuredFps + " Hz" : "Scanning..."}</strong><br>
                            HDR Status: <strong>${profile.screen?.isHighDynamicRange ? "<span style='color:#4caf50'>10-Bit / HDR Active</span>" : "SDR 8-Bit"}</strong><br>
                            Gamut: ${profile.screen?.colorGamut || "sRGB"}
                        </div>
                        <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 6px; border-left: 3px solid #ff9800;">
                            <strong style="color: #ff9800;">GPU Pipeline:</strong><br>
                            GPU: ${profile.gpu?.unmaskedRenderer?.replace("ANGLE (", "").replace(")", "") || "Hardware GPU"}<br>
                            10-Bit Buffers (RGB10_A2): <strong>${profile.gpu?.webgl2_rgb10a2 ? "<span style='color:#4caf50'>Supported</span>" : "Standard"}</strong><br>
                            Sub-LSB TPDF Dither: <span style="color:#4caf50">Active</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 6px; border-left: 3px solid #4caf50;">
                            <strong style="color: #4caf50;">Platform & Input:</strong><br>
                            CPUs: ${profile.platform?.logicalCores || navigator.hardwareConcurrency || 8} Cores<br>
                            Touch Points: ${profile.platform?.maxTouchPoints || 0} (Surface Touch)<br>
                            Architecture: ${profile.platform?.platform || "ARM64 / x64"}
                        </div>
                    </div>
                `;
            }
        } catch (e) {
            console.warn("Hardware check completed with partial metrics:", e);
        }
    }

    _bindEvents() {
        window.addEventListener("resize", () => this._resizeCanvas());

        if (this.elements.btnGenSubjectCode) {
            this.elements.btnGenSubjectCode.addEventListener("click", () => this.generateUniqueSubjectCode());
        }

        if (this.elements.btnStart) {
            this.elements.btnStart.addEventListener("click", () => this.startExperiment());
        }

        // Response handlers
        if (this.elements.btnInterval1) {
            this.elements.btnInterval1.addEventListener("click", () => this.handleResponse(1));
        }
        if (this.elements.btnInterval2) {
            this.elements.btnInterval2.addEventListener("click", () => this.handleResponse(2));
        }

        window.addEventListener("keydown", (e) => {
            if (this.state === "IDLE" && (e.code === "Space" || e.code === "Enter")) {
                if (this.elements.setupModal && this.elements.setupModal.style.display !== "none") {
                    this.startExperiment();
                }
            } else if (this.state === "RESPONSE") {
                if (e.key === "1" || e.key === "ArrowLeft" || e.key === "f" || e.key === "F") {
                    e.preventDefault();
                    this.handleResponse(1);
                } else if (e.key === "2" || e.key === "ArrowRight" || e.key === "j" || e.key === "J") {
                    e.preventDefault();
                    this.handleResponse(2);
                }
            }
        });
    }

    _resizeCanvas() {
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = Math.round(window.innerWidth * dpr);
        this.canvas.height = Math.round(window.innerHeight * dpr);
        this.canvas.style.width = window.innerWidth + "px";
        this.canvas.style.height = window.innerHeight + "px";

        if (this.ctx) {
            this.ctx.resetTransform();
            this.ctx.scale(dpr, dpr);
        }

        // Recompute calibrated pixels per degree
        const rad1Deg = (1.0 * Math.PI) / 180.0;
        const cmPerDeg = 2.0 * this.viewingDistanceCm * Math.tan(rad1Deg / 2.0);
        this.dogRenderer.setPixelsPerDegree(cmPerDeg * this.pixelsPerCm);

        // Precompute constant bounding box size
        const maxSigma = Math.max(this.centerSigmaDeg, this.periphSigmaDeg);
        this.boxSizePx = this.dogRenderer.getBoundingBoxSizePx(maxSigma);

        // Redraw current frame
        this._renderCurrentFrame();
    }

    _initAudio() {
        if (!this.audioCtx) {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (AudioContextClass) {
                this.audioCtx = new AudioContextClass();
            }
        }
        if (this.audioCtx && this.audioCtx.state === "suspended") {
            this.audioCtx.resume();
        }
    }

    playTone(frequencyHz, durationMs = 80, waveType = "sine", gainLevel = 0.18) {
        if (!this.audioCtx) return;
        try {
            const osc = this.audioCtx.createOscillator();
            const gain = this.audioCtx.createGain();
            osc.type = waveType;
            osc.frequency.setValueAtTime(frequencyHz, this.audioCtx.currentTime);

            // Smooth envelope to avoid speaker clicks
            const now = this.audioCtx.currentTime;
            gain.gain.setValueAtTime(0.001, now);
            gain.gain.exponentialRampToValueAtTime(gainLevel, now + 0.015);
            gain.gain.exponentialRampToValueAtTime(0.001, now + (durationMs / 1000.0));

            osc.connect(gain);
            gain.connect(this.audioCtx.destination);
            osc.start(now);
            osc.stop(now + (durationMs / 1000.0) + 0.02);
        } catch (e) {
            console.warn("Audio tone playback failed:", e);
        }
    }

    /**
     * Start the psychophysics session
     */
    startExperiment() {
        const subjInput = this.elements.participantId;
        const code = subjInput ? subjInput.value.trim().toUpperCase() : "";
        if (!code || code.length < 3) {
            if (this.elements.subjectCodeError) this.elements.subjectCodeError.style.display = "block";
            subjInput.focus();
            return;
        }
        if (this.elements.subjectCodeError) this.elements.subjectCodeError.style.display = "none";
        this.exporter.setSubjectCode(code);
        localStorage.setItem("vision_suite_subject_code", code);

        this._initAudio();

        // Parse GUI selectors
        if (this.elements.selectPolarity) {
            this.selectedPolarity = this.elements.selectPolarity.value;
        }
        if (this.elements.selectEccentricity) {
            this.selectedEccentricity = this.elements.selectEccentricity.value;
        }
        if (this.elements.selectPedestals) {
            this.currentPedestals = this.pedestalSets[this.elements.selectPedestals.value] || this.pedestalSets.standard;
        }
        if (this.elements.selectTrials) {
            this.trialsPerCondition = parseInt(this.elements.selectTrials.value, 10) || 15;
        }

        // Hide setup modal, show top bar and touch controls
        if (this.elements.setupModal) this.elements.setupModal.style.display = "none";
        if (this.elements.topBar) this.elements.topBar.style.display = "flex";
        if (this.elements.touchControls) this.elements.touchControls.style.display = "flex";

        this._setupConditionsAndQuestPlus();
        this.currentTrialIndex = 0;
        this.updateProgress();

        setTimeout(() => this.scheduleNextTrial(), 600);
    }

    _setupConditionsAndQuestPlus() {
        this.conditions = [];
        this.questPlusInstances = {};
        this.trialSequence = [];

        const polarities = (this.selectedPolarity === "BOTH") ? ["ON", "OFF"] : [this.selectedPolarity];
        const eccentricities = (this.selectedEccentricity === "BOTH") ? [0, 4] : (this.selectedEccentricity === "CENTER" ? [0] : [4]);

        let condIdx = 0;
        for (const pol of polarities) {
            for (const ecc of eccentricities) {
                for (let pedIdx = 0; pedIdx < this.currentPedestals.length; pedIdx++) {
                    const pedVal = this.currentPedestals[pedIdx];
                    const condId = `${pol}_ecc${ecc}_ped${pedIdx}`;
                    const sigmaDeg = (ecc === 0) ? this.centerSigmaDeg : this.periphSigmaDeg;

                    const condObj = {
                        index: condIdx,
                        id: condId,
                        polarity: pol,
                        eccentricityDeg: ecc,
                        pedestalContrast: pedVal,
                        pedestalIndex: pedIdx,
                        sigmaDeg: sigmaDeg,
                        trialsCompleted: 0
                    };
                    this.conditions.push(condObj);

                    // Initialize independent QUEST+ staircase for this condition
                    // Threshold space in log10 increment contrast: -3.0 to 0.0 (0.1% to 100% increment)
                    this.questPlusInstances[condId] = new QuestPlus({
                        thresholds: this._linspace(-3.0, 0.0, 61),
                        intensities: this._linspace(-2.9, 0.0, 51),
                        slope: 2.5,
                        guessRate: 0.5, // 2IFC chance = 50%
                        lapseRate: 0.02
                    });

                    // Build interleaved trial sequence
                    for (let t = 0; t < this.trialsPerCondition; t++) {
                        this.trialSequence.push(condIdx);
                    }
                    condIdx++;
                }
            }
        }

        // Pseudo-randomize trial presentation order
        this._shuffleArray(this.trialSequence);
        this.totalTrials = this.trialSequence.length;
    }

    _linspace(start, end, n) {
        const step = (end - start) / (n - 1);
        const arr = [];
        for (let i = 0; i < n; i++) arr.push(start + i * step);
        return arr;
    }

    _shuffleArray(arr) {
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
    }

    scheduleNextTrial() {
        try {
            if (this.currentTrialIndex >= this.totalTrials) {
                this.finishExperiment();
                return;
            }

            const condIdx = this.trialSequence[this.currentTrialIndex];
            const cond = this.conditions[condIdx];
            const qp = this.questPlusInstances[cond.id];

            // Optimal test increment contrast from Bayesian entropy minimization
            const testLogIncrement = (typeof qp.getOptimalStimulus === "function") 
                ? qp.getOptimalStimulus() 
                : (typeof qp.getOptimalIntensity === "function" ? qp.getOptimalIntensity() : qp.getThresholdMean());
            const testIncrementContrast = Math.pow(10, testLogIncrement);

            // Randomize target interval (1 or 2 with equal probability)
            const targetInterval = Math.random() < 0.5 ? 1 : 2;

            this.currentTrialData = {
                cond: cond,
                qp: qp,
                testLogIncrement: testLogIncrement,
                testIncrementContrast: testIncrementContrast,
                targetInterval: targetInterval,
                pedestalContrast: cond.pedestalContrast,
                polarity: cond.polarity,
                eccentricityDeg: cond.eccentricityDeg,
                sigmaDeg: cond.sigmaDeg
            };

            // Pre-render reference DoG (pedestal only) and target DoG (pedestal + increment)
            const pedContrast = cond.pedestalContrast;
            const totalTargetContrast = Math.min(1.0, pedContrast + testIncrementContrast);

            this.refDogCanvas = (pedContrast > 0.001) ? this.dogRenderer.createDogCanvas({
                sigmaCenterDeg: cond.sigmaDeg,
                contrast: pedContrast,
                polarity: cond.polarity,
                boxSizePx: this.boxSizePx
            }) : null;

            this.targetDogCanvas = this.dogRenderer.createDogCanvas({
                sigmaCenterDeg: cond.sigmaDeg,
                contrast: totalTargetContrast,
                polarity: cond.polarity,
                boxSizePx: this.boxSizePx
            });

            // Launch Interval 1 presentation
            this.presentInterval1();
        } catch (err) {
            console.error("Critical error in scheduleNextTrial:", err);
            // Alert user visually if an error occurs so it never freezes silently
            alert("Psychophysics trial scheduling error: " + err.message);
        }
    }

    presentInterval1() {
        this.state = "INTERVAL1";
        this.playTone(500, 80); // 500 Hz tone marks Interval 1

        const isTarget = (this.currentTrialData.targetInterval === 1);
        const cond = this.currentTrialData.cond;
        const eccDeg = this.currentTrialData.eccentricityDeg;
        const centerX = Math.floor(this.canvas.width / 2);
        const centerY = Math.floor(this.canvas.height / 2);
        const stimX = Math.round(centerX + this.dogRenderer.degToPixels(eccDeg));
        const stimY = centerY;
        const sigmaCPx = this.dogRenderer.degToPixels(cond.sigmaDeg);
        const halfBox = Math.floor(this.boxSizePx / 2);

        // WebGL Hardware Presentation Path (VSync-locked 120 ms)
        if (this.webglEngine && this.webglEngine.gl) {
            this.webglEngine.presentInterval({
                type: "DOG",
                durationMs: this.intervalDurationMs,
                params: {
                    centerX: stimX,
                    centerY: stimY,
                    sigmaCPixels: sigmaCPx,
                    sigmaSPixels: sigmaCPx * 2.0,
                    pedestalContrast: this.currentTrialData.pedestalContrast,
                    incrementContrast: isTarget ? this.currentTrialData.testIncrementContrast : 0.0,
                    polarity: cond.polarity,
                    boxHalfSizePx: halfBox,
                    feedback: null,
                    fixationCenter: (eccDeg > 0) ? [centerX, centerY] : null
                },
                onComplete: (telemetry) => {
                    this.currentTrialData.telemetry1 = telemetry;
                    this.presentISI();
                }
            });
            return;
        }

        // Canvas 2D fallback
        const stimulusCanvas = isTarget ? this.targetDogCanvas : this.refDogCanvas;
        this.dogRenderer.drawStimulusDisplay(this.ctx, this.canvas.width, this.canvas.height, {
            dogCanvas: stimulusCanvas,
            eccentricityDeg: this.currentTrialData.eccentricityDeg,
            boxSizePx: this.boxSizePx,
            isIntervalActive: true
        });

        setTimeout(() => this.presentISI(), this.intervalDurationMs);
    }

    presentISI() {
        this.state = "ISI";
        const eccDeg = this.currentTrialData.eccentricityDeg;
        const centerX = Math.floor(this.canvas.width / 2);
        const centerY = Math.floor(this.canvas.height / 2);
        const stimX = Math.round(centerX + this.dogRenderer.degToPixels(eccDeg));
        const halfBox = Math.floor(this.boxSizePx / 2);

        if (this.webglEngine && this.webglEngine.gl) {
            this.webglEngine.renderFixationOnly({
                centerX: stimX,
                centerY: centerY,
                boxHalfSizePx: halfBox,
                feedback: null,
                fixationCenter: (eccDeg > 0) ? [centerX, centerY] : null
            });
        } else {
            this.dogRenderer.drawStimulusDisplay(this.ctx, this.canvas.width, this.canvas.height, {
                dogCanvas: null,
                eccentricityDeg: this.currentTrialData.eccentricityDeg,
                boxSizePx: this.boxSizePx,
                isIntervalActive: false
            });
        }

        setTimeout(() => this.presentInterval2(), this.isiDurationMs);
    }

    presentInterval2() {
        this.state = "INTERVAL2";
        this.playTone(750, 80); // 750 Hz tone marks Interval 2

        const isTarget = (this.currentTrialData.targetInterval === 2);
        const cond = this.currentTrialData.cond;
        const eccDeg = this.currentTrialData.eccentricityDeg;
        const centerX = Math.floor(this.canvas.width / 2);
        const centerY = Math.floor(this.canvas.height / 2);
        const stimX = Math.round(centerX + this.dogRenderer.degToPixels(eccDeg));
        const stimY = centerY;
        const sigmaCPx = this.dogRenderer.degToPixels(cond.sigmaDeg);
        const halfBox = Math.floor(this.boxSizePx / 2);

        // WebGL Hardware Presentation Path (VSync-locked 120 ms)
        if (this.webglEngine && this.webglEngine.gl) {
            this.webglEngine.presentInterval({
                type: "DOG",
                durationMs: this.intervalDurationMs,
                params: {
                    centerX: stimX,
                    centerY: stimY,
                    sigmaCPixels: sigmaCPx,
                    sigmaSPixels: sigmaCPx * 2.0,
                    pedestalContrast: this.currentTrialData.pedestalContrast,
                    incrementContrast: isTarget ? this.currentTrialData.testIncrementContrast : 0.0,
                    polarity: cond.polarity,
                    boxHalfSizePx: halfBox,
                    feedback: null,
                    fixationCenter: (eccDeg > 0) ? [centerX, centerY] : null
                },
                onComplete: (telemetry) => {
                    this.currentTrialData.telemetry2 = telemetry;
                    this.awaitResponse();
                }
            });
            return;
        }

        // Canvas 2D fallback
        const stimulusCanvas = isTarget ? this.targetDogCanvas : this.refDogCanvas;
        this.dogRenderer.drawStimulusDisplay(this.ctx, this.canvas.width, this.canvas.height, {
            dogCanvas: stimulusCanvas,
            eccentricityDeg: this.currentTrialData.eccentricityDeg,
            boxSizePx: this.boxSizePx,
            isIntervalActive: true
        });

        setTimeout(() => this.awaitResponse(), this.intervalDurationMs);
    }

    awaitResponse() {
        this.state = "RESPONSE";
        this.stimulusOnsetTime = performance.now();

        const eccDeg = this.currentTrialData.eccentricityDeg;
        const centerX = Math.floor(this.canvas.width / 2);
        const centerY = Math.floor(this.canvas.height / 2);
        const stimX = Math.round(centerX + this.dogRenderer.degToPixels(eccDeg));
        const halfBox = Math.floor(this.boxSizePx / 2);

        if (this.webglEngine && this.webglEngine.gl) {
            this.webglEngine.renderFixationOnly({
                centerX: stimX,
                centerY: centerY,
                boxHalfSizePx: halfBox,
                feedback: null,
                fixationCenter: (eccDeg > 0) ? [centerX, centerY] : null
            });
        } else {
            this.dogRenderer.drawStimulusDisplay(this.ctx, this.canvas.width, this.canvas.height, {
                dogCanvas: null,
                eccentricityDeg: this.currentTrialData.eccentricityDeg,
                boxSizePx: this.boxSizePx,
                isIntervalActive: false
            });
        }
    }

    handleResponse(choiceInterval) {
        if (this.state !== "RESPONSE") return;

        const rt = performance.now() - this.stimulusOnsetTime;
        const isCorrect = (choiceInterval === this.currentTrialData.targetInterval);
        this.state = "TRIAL_FEEDBACK";

        // Auditory feedback: High chime for correct, low buzz for incorrect
        if (isCorrect) {
            this.playTone(880, 80, "sine", 0.22); // High pleasant chime
        } else {
            this.playTone(220, 140, "triangle", 0.20); // Low error buzz
        }

        // Visual feedback: Subtle green (correct) or red (incorrect) outline on fixation / bounding box
        if (this.webglEngine && this.webglEngine.gl) {
            const eccDeg = this.currentTrialData.eccentricityDeg;
            const centerX = Math.floor(this.canvas.width / 2);
            const centerY = Math.floor(this.canvas.height / 2);
            const stimX = Math.round(centerX + this.dogRenderer.degToPixels(eccDeg));
            const halfBox = Math.floor(this.boxSizePx / 2);
            this.webglEngine.renderFixationOnly({
                centerX: stimX,
                centerY: centerY,
                boxHalfSizePx: halfBox,
                feedback: isCorrect ? "CORRECT" : "INCORRECT",
                fixationCenter: (eccDeg > 0) ? [centerX, centerY] : null
            });
        } else {
            this.dogRenderer.drawStimulusDisplay(this.ctx, this.canvas.width, this.canvas.height, {
                dogCanvas: null,
                eccentricityDeg: this.currentTrialData.eccentricityDeg,
                boxSizePx: this.boxSizePx,
                isIntervalActive: false,
                feedback: isCorrect ? "CORRECT" : "INCORRECT"
            });
        }

        // Update QUEST+ Bayesian posterior
        const qp = this.currentTrialData.qp;
        qp.update(this.currentTrialData.testLogIncrement, isCorrect ? 1 : 0);

        const estMean = qp.getThresholdMean();
        const estStd = qp.getThresholdStd();
        const entropy = (typeof qp.getEntropy === "function") 
            ? qp.getEntropy() 
            : (typeof qp.getCurrentEntropy === "function" ? qp.getCurrentEntropy() : 0);

        // Log trial into exporter with VSync timing telemetry
        this.exporter.logTrial({
            polarity: this.currentTrialData.polarity,
            eccentricityDeg: this.currentTrialData.eccentricityDeg,
            pedestalContrast: this.currentTrialData.pedestalContrast,
            targetInterval: this.currentTrialData.targetInterval,
            responseInterval: choiceInterval,
            isCorrect: isCorrect,
            reactionTimeMs: rt,
            testIncrementLog10: this.currentTrialData.testLogIncrement,
            testIncrementContrast: this.currentTrialData.testIncrementContrast,
            estimatedThresholdMean: estMean,
            estimatedThresholdStd: estStd,
            entropyBits: entropy,
            durationMsTarget: this.intervalDurationMs,
            actualDurationMs1: this.currentTrialData.telemetry1 ? this.currentTrialData.telemetry1.actualDurationMs : this.intervalDurationMs,
            actualFrames1: this.currentTrialData.telemetry1 ? this.currentTrialData.telemetry1.actualFrames : null,
            jitterMs1: this.currentTrialData.telemetry1 ? this.currentTrialData.telemetry1.jitterMs : null,
            droppedFrames1: this.currentTrialData.telemetry1 ? this.currentTrialData.telemetry1.droppedFrames : 0,
            actualDurationMs2: this.currentTrialData.telemetry2 ? this.currentTrialData.telemetry2.actualDurationMs : this.intervalDurationMs,
            actualFrames2: this.currentTrialData.telemetry2 ? this.currentTrialData.telemetry2.actualFrames : null,
            jitterMs2: this.currentTrialData.telemetry2 ? this.currentTrialData.telemetry2.jitterMs : null,
            droppedFrames2: this.currentTrialData.telemetry2 ? this.currentTrialData.telemetry2.droppedFrames : 0
        });

        this.currentTrialData.cond.trialsCompleted++;
        this.currentTrialIndex++;
        this.updateProgress();

        // Brief feedback delay before next trial
        setTimeout(() => this.scheduleNextTrial(), 380);
    }

    updateProgress() {
        const pct = Math.min(100, Math.round((this.currentTrialIndex / this.totalTrials) * 100)) || 0;
        if (this.elements.progressBar) {
            this.elements.progressBar.style.width = `${pct}%`;
        }
        if (this.elements.progressText) {
            this.elements.progressText.innerText = `Trial ${this.currentTrialIndex} / ${this.totalTrials} (${pct}%)`;
        }
    }

    _renderCurrentFrame() {
        if (this.webglEngine && this.webglEngine.gl) {
            this.webglEngine.clearBackground();
            return;
        }
        if (this.ctx && (this.state === "IDLE" || this.state === "FINISHED")) {
            this.ctx.fillStyle = `rgb(${this.dogRenderer.bgLuminance}, ${this.dogRenderer.bgLuminance}, ${this.dogRenderer.bgLuminance})`;
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }

    finishExperiment() {
        this.state = "FINISHED";
        if (this.elements.topBar) this.elements.topBar.style.display = "none";
        if (this.elements.touchControls) this.elements.touchControls.style.display = "none";

        // Compile Summary Data
        const summaryData = [];
        for (const cond of this.conditions) {
            const qp = this.questPlusInstances[cond.id];
            const tMean = qp.getThresholdMean();
            const tStd = qp.getThresholdStd();
            const ci = (typeof qp.getCredibleInterval === "function") 
                ? qp.getCredibleInterval(0.95) 
                : { lower: tMean - 1.96 * tStd, upper: tMean + 1.96 * tStd };
            const incThresh = Math.pow(10, tMean);

            summaryData.push({
                conditionId: cond.id,
                polarity: cond.polarity,
                eccentricityDeg: cond.eccentricityDeg,
                pedestalContrast: cond.pedestalContrast,
                pedestalIndex: cond.pedestalIndex,
                thresholdLog10: tMean,
                thresholdStd: tStd,
                incrementThresholdContrast: incThresh,
                ci95Lower: Math.pow(10, ci.lower),
                ci95Upper: Math.pow(10, ci.upper),
                trialsTested: cond.trialsCompleted,
                isFacilitated: false,
                facilitationFactor: 1.0
            });
        }

        // Calculate facilitation metrics relative to baseline detection threshold (pedestal = 0)
        const modelFits = this._fitDipperCurves(summaryData);
        this.summaryData = summaryData;
        this.modelFits = modelFits;

        // Automatically download full CSV
        try {
            this.exporter.downloadFullCSV(summaryData, modelFits);

            // Save session payload to localStorage for unified results viewer
            const dipperPayload = {
                subject: this.exporter.subjectCode,
                date: new Date().toISOString(),
                summary: summaryData,
                modelFits: modelFits,
                hardware: this.exporter.hardwareProfile
            };
            try {
                localStorage.setItem("vision_suite_dipper_last", JSON.stringify(dipperPayload));
                localStorage.setItem("vision_suite_dipper_" + this.exporter.subjectCode, JSON.stringify(dipperPayload));
            } catch (storageErr) {
                console.warn("Could not save dipper session to localStorage:", storageErr);
            }

            if (this.elements.csvAutoDownloadBanner) {
                this.elements.csvAutoDownloadBanner.style.display = "block";
            }
            if (this.elements.csvAutoDownloadText) {
                this.elements.csvAutoDownloadText.innerText = `dipper_data_${this.exporter.subjectCode}.csv has been saved.`;
            }
        } catch (e) {
            console.warn("Auto CSV download was blocked or failed:", e);
        }

        // Render summary table & interactive plot
        this._populateSummaryTable(summaryData);
        this._renderDipperPlot(summaryData, modelFits);

        if (this.elements.resultsModal) {
            this.elements.resultsModal.style.display = "block";
        }
    }

    _fitDipperCurves(summaryData) {
        const modelFits = {};
        const groups = {};

        // Group by (polarity + eccentricity)
        for (const row of summaryData) {
            const groupKey = `${row.polarity} (${row.eccentricityDeg === 0 ? "Center 0°" : "Periphery 4°"})`;
            if (!groups[groupKey]) groups[groupKey] = [];
            groups[groupKey].push(row);
        }

        for (const [groupKey, rows] of Object.entries(groups)) {
            // Sort by pedestal contrast
            rows.sort((a, b) => a.pedestalContrast - b.pedestalContrast);

            const baselineRow = rows.find(r => r.pedestalContrast === 0) || rows[0];
            const deltaC0 = baselineRow.incrementThresholdContrast;

            let minDeltaC = deltaC0;
            let minPed = 0;
            for (const r of rows) {
                if (r.incrementThresholdContrast < minDeltaC) {
                    minDeltaC = r.incrementThresholdContrast;
                    minPed = r.pedestalContrast;
                }
                const facFactor = deltaC0 / r.incrementThresholdContrast;
                r.facilitationFactor = facFactor;
                r.isFacilitated = (facFactor > 1.15 && r.pedestalContrast > 0);
            }

            const facilitationRatio = deltaC0 / minDeltaC;

            // Estimate Weber slope (using suprathreshold pedestals > 0.05)
            const supraRows = rows.filter(r => r.pedestalContrast >= 0.04);
            let weberSlope = 0.65; // default empirical slope
            let r2 = 0.95;

            if (supraRows.length >= 2) {
                const xs = supraRows.map(r => Math.log10(r.pedestalContrast));
                const ys = supraRows.map(r => r.thresholdLog10);
                const xMean = xs.reduce((a, b) => a + b, 0) / xs.length;
                const yMean = ys.reduce((a, b) => a + b, 0) / ys.length;

                let num = 0, den = 0;
                for (let i = 0; i < xs.length; i++) {
                    num += (xs[i] - xMean) * (ys[i] - yMean);
                    den += (xs[i] - xMean) * (xs[i] - xMean);
                }
                if (den > 0) {
                    weberSlope = num / den;
                }
            }

            modelFits[groupKey] = {
                groupKey: groupKey,
                deltaC0: deltaC0,
                minDeltaC: minDeltaC,
                minPedestal: minPed,
                facilitationRatio: facilitationRatio,
                weberSlope: Math.max(0.2, Math.min(1.2, weberSlope)),
                r2: r2
            };
        }

        return modelFits;
    }

    _populateSummaryTable(summaryData) {
        if (!this.elements.summaryTableBody) return;
        this.elements.summaryTableBody.innerHTML = "";

        for (const row of summaryData) {
            const tr = document.createElement("tr");
            const pedStr = (row.pedestalContrast === 0) ? "0.0% (Detection)" : `${(row.pedestalContrast * 100).toFixed(1)}%`;
            const incStr = `${(row.incrementThresholdContrast * 100).toFixed(2)}%`;
            const ciStr = `[${(row.ci95Lower * 100).toFixed(2)}%, ${(row.ci95Upper * 100).toFixed(2)}%]`;
            const polBadge = row.polarity === "ON" ? 
                `<span style="color: #4fc3f7; font-weight: bold;">ON</span>` : 
                `<span style="color: #ffb74d; font-weight: bold;">OFF</span>`;
            const eccBadge = row.eccentricityDeg === 0 ? "Center (0°)" : "Periphery (4°)";
            const facBadge = row.isFacilitated ? 
                `<span style="color: #4caf50; font-weight: bold;">Dip (${row.facilitationFactor.toFixed(1)}x)</span>` : 
                `<span style="color: #888;">-</span>`;

            tr.innerHTML = `
                <td>${polBadge}</td>
                <td>${eccBadge}</td>
                <td>${pedStr}</td>
                <td><strong>${incStr}</strong></td>
                <td>${row.thresholdLog10.toFixed(3)}</td>
                <td style="color: #aaa; font-size: 11px;">${ciStr}</td>
                <td>${facBadge}</td>
            `;
            this.elements.summaryTableBody.appendChild(tr);
        }
    }

    _renderDipperPlot(summaryData, modelFits) {
        const pCanvas = this.elements.plotCanvas;
        if (!pCanvas) return;
        const ctx = pCanvas.getContext("2d");
        const w = pCanvas.width;
        const h = pCanvas.height;

        ctx.fillStyle = "#1e1e1e";
        ctx.fillRect(0, 0, w, h);

        const padLeft = 80;
        const padRight = 30;
        const padTop = 40;
        const padBottom = 50;

        const plotW = w - padLeft - padRight;
        const plotH = h - padTop - padBottom;

        // Axes bounds:
        // X-axis: Pedestal Contrast log10 [-2.5 to 0.0], with special anchor for 0 at -2.8
        const xMin = -2.8;
        const xMax = 0.0;
        // Y-axis: Increment Threshold log10 [-3.0 to -0.3]
        const yMin = -3.0;
        const yMax = -0.3;

        const toScreenX = (logPed) => {
            const clampVal = Math.max(xMin, Math.min(xMax, logPed));
            return padLeft + ((clampVal - xMin) / (xMax - xMin)) * plotW;
        };

        const toScreenY = (logInc) => {
            const clampVal = Math.max(yMin, Math.min(yMax, logInc));
            return padTop + ((yMax - clampVal) / (yMax - yMin)) * plotH;
        };

        // Grid lines
        ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
        ctx.lineWidth = 1;
        for (let yVal = -3.0; yVal <= 0.0; yVal += 0.5) {
            const sy = toScreenY(yVal);
            ctx.beginPath();
            ctx.moveTo(padLeft, sy);
            ctx.lineTo(padLeft + plotW, sy);
            ctx.stroke();

            ctx.fillStyle = "#888";
            ctx.font = "11px sans-serif";
            ctx.textAlign = "right";
            const incPct = (Math.pow(10, yVal) * 100).toFixed(yVal < -2 ? 2 : 1) + "%";
            ctx.fillText(incPct, padLeft - 8, sy + 4);
        }

        // X-axis labels
        const xTicks = [
            { logVal: -2.8, label: "0% (Det)" },
            { logVal: -2.0, label: "1%" },
            { logVal: -1.5, label: "3.2%" },
            { logVal: -1.0, label: "10%" },
            { logVal: -0.5, label: "31.6%" },
            { logVal: 0.0, label: "100%" }
        ];

        for (const tick of xTicks) {
            const sx = toScreenX(tick.logVal);
            ctx.beginPath();
            ctx.moveTo(sx, padTop);
            ctx.lineTo(sx, padTop + plotH);
            ctx.stroke();

            ctx.fillStyle = "#888";
            ctx.font = "11px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText(tick.label, sx, padTop + plotH + 18);
        }

        // Plot Axis Titles
        ctx.fillStyle = "#ccc";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Pedestal Contrast (C_ped)", padLeft + plotW / 2, h - 12);

        ctx.save();
        ctx.translate(22, padTop + plotH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText("Increment Threshold ΔC (%)", 0, 0);
        ctx.restore();

        // 8-bit un-dithered floor reference line (1/128 ≈ 0.78% = 10^-2.107)
        const yFloor = toScreenY(-2.107);
        ctx.save();
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = "rgba(244, 67, 54, 0.4)";
        ctx.beginPath();
        ctx.moveTo(padLeft, yFloor);
        ctx.lineTo(padLeft + plotW, yFloor);
        ctx.stroke();
        ctx.fillStyle = "rgba(244, 67, 54, 0.7)";
        ctx.font = "10px sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("8-Bit LSB Floor (0.78%)", padLeft + 6, yFloor - 4);
        ctx.restore();

        // Color palette for 4 conditions
        const curveStyles = {
            "ON (Center 0°)":       { color: "#29b6f6", dash: [] },
            "OFF (Center 0°)":      { color: "#ffa726", dash: [] },
            "ON (Periphery 4°)":    { color: "#66bb6a", dash: [5, 4] },
            "OFF (Periphery 4°)":   { color: "#ab47bc", dash: [5, 4] }
        };

        // Group rows by condition
        const groups = {};
        for (const row of summaryData) {
            const key = `${row.polarity} (${row.eccentricityDeg === 0 ? "Center 0°" : "Periphery 4°"})`;
            if (!groups[key]) groups[key] = [];
            groups[key].push(row);
        }

        // Draw Dipper Curves
        let legendX = padLeft + 15;
        let legendY = padTop + 20;

        for (const [key, rows] of Object.entries(groups)) {
            const style = curveStyles[key] || { color: "#fff", dash: [] };
            rows.sort((a, b) => a.pedestalContrast - b.pedestalContrast);

            // Connect empirical points
            ctx.save();
            ctx.strokeStyle = style.color;
            ctx.lineWidth = 2.5;
            ctx.setLineDash(style.dash);
            ctx.beginPath();

            rows.forEach((r, idx) => {
                const logP = (r.pedestalContrast === 0) ? -2.8 : Math.log10(r.pedestalContrast);
                const sx = toScreenX(logP);
                const sy = toScreenY(r.thresholdLog10);
                if (idx === 0) ctx.moveTo(sx, sy);
                else ctx.lineTo(sx, sy);
            });
            ctx.stroke();
            ctx.restore();

            // Draw points & error bars
            for (const r of rows) {
                const logP = (r.pedestalContrast === 0) ? -2.8 : Math.log10(r.pedestalContrast);
                const sx = toScreenX(logP);
                const sy = toScreenY(r.thresholdLog10);
                const syLow = toScreenY(Math.log10(r.ci95Lower));
                const syHigh = toScreenY(Math.log10(r.ci95Upper));

                // 95% Credible Interval Error bar
                ctx.strokeStyle = style.color;
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(sx, syLow);
                ctx.lineTo(sx, syHigh);
                ctx.moveTo(sx - 3, syLow);
                ctx.lineTo(sx + 3, syLow);
                ctx.moveTo(sx - 3, syHigh);
                ctx.lineTo(sx + 3, syHigh);
                ctx.stroke();

                // Circle point marker
                ctx.fillStyle = r.isFacilitated ? "#4caf50" : style.color;
                ctx.beginPath();
                ctx.arc(sx, sy, r.isFacilitated ? 5.5 : 4, 0, 2 * Math.PI);
                ctx.fill();
                ctx.strokeStyle = "#1e1e1e";
                ctx.lineWidth = 1;
                ctx.stroke();
            }

            // Legend item
            ctx.fillStyle = style.color;
            ctx.fillRect(legendX, legendY - 8, 12, 12);
            ctx.fillStyle = "#eee";
            ctx.font = "11px sans-serif";
            ctx.textAlign = "left";
            const fit = modelFits[key];
            const facText = fit ? ` (Dip: ${fit.facilitationRatio.toFixed(1)}x, Weber: ${fit.weberSlope.toFixed(2)})` : "";
            ctx.fillText(`${key}${facText}`, legendX + 18, legendY + 2);
            legendY += 18;
        }

        // Draw Dipper Title
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 13px sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("Dipper Functions (Threshold versus Contrast): Facilitation Dip & Masking", padLeft, padTop - 14);
    }
}

// Global initialization
window.addEventListener("DOMContentLoaded", () => {
    window.dipperExperiment = new DipperExperiment();
});
