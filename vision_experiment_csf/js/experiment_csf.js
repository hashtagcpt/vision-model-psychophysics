/**
 * experiment_csf.js - Fullscreen Two-Interval Forced Choice (2IFC 2AFC) Contrast Sensitivity Function
 * 
 * Features:
 * - Fullscreen neutral mid-gray window (rgb(128, 128, 128)) edge-to-edge
 * - 10-Bit & HDR Hardware Diagnostic Suite (auto-detects Surface Pro OLED wide-gamut pipeline)
 * - Constant Gabor envelope size (default sigma = 0.5 deg visual angle) across all spatial frequencies
 * - Two-Interval Forced Choice (2IFC 2AFC) detection task with dual auditory interval beeps:
 *     Interval 1: Beep 1 (500 Hz, 80 ms) -> 200 ms window
 *     ISI:        400 ms blank mid-gray
 *     Interval 2: Beep 2 (750 Hz, 80 ms) -> 200 ms window
 * - Strict 2AFC chance level (gamma = 0.5) eliminating observer criterion bias
 * - Interleaved QUEST+ Bayesian adaptive staircases across 6 spatial frequencies [0.5 to 16 cpd]
 * - Display Bit-Depth & Spatial Noise Dithering (sub-LSB contrast preservation)
 * - High-DPI Canvas Rendering of final Double-Logarithmic CSF Plot with model fit & 8-bit floor
 */

class CSFExperiment {
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

        this.plotCanvas = document.getElementById("csfPlotCanvas");
        this.plotCtx = this.plotCanvas ? this.plotCanvas.getContext("2d") : null;

        this.exporter = new CSFExporter();
        this.hardwareChecker = new HardwareChecker();
        this.renderer = null;

        // Experiment Configuration: Full 8-Octave Ultra-Wide Range (0.125 to 32 cpd)
        this.spatialFrequencies = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
        this.trialsPerCondition = 20; // 20 trials per SF
        this.sigmaDeg = 0.5; // Base constant envelope size across all spatial frequencies
        this.viewingDistanceCm = 57.3;
        this.pixelsPerCm = 38.0;

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
                    const distInput = document.getElementById("viewingDistance");
                    if (distInput) distInput.value = data.viewingDistanceCm;
                    this._resizeCanvas();
                };
            }
        }
        this.bitDepth = 8;
        this.enableDithering = true;
        this.participantId = "Observer_01";

        // Timing parameters (in ms) - 120 ms stimulus duration
        this.preFixationDuration = 400;
        this.intervalDuration = 120;
        this.interStimulusInterval = 400; // ISI between Interval 1 and Interval 2
        this.interTrialInterval = 500;

        // State Machine:
        // IDLE, FIXATION_PRE, INTERVAL_1, ISI, INTERVAL_2, AWAIT_RESPONSE, ITI, COMPLETED
        this.state = "IDLE";
        this.trialQueue = [];
        this.currentTrialIndex = 0;
        this.currentTrialData = null;
        this.questInstances = new Map(); // SF -> QuestPlus
        this.responseWindowStartTime = 0;
        this.timerId = null;

        // Sound cues using Web Audio API
        this.audioCtx = null;

        // UI references
        this.ui = {
            screenSetup: document.getElementById("screenSetup"),
            screenExperiment: document.getElementById("screenExperiment"),
            screenResults: document.getElementById("screenResults"),
            hardwareReportContainer: document.getElementById("hardwareReportContainer"),
            participantInput: document.getElementById("participantId"),
            btnGenSubjectCode: document.getElementById("btnGenSubjectCode"),
            subjectCodeError: document.getElementById("subjectCodeError"),
            csvAutoDownloadBanner: document.getElementById("csvAutoDownloadBanner"),
            csvAutoDownloadText: document.getElementById("csvAutoDownloadText"),
            hudProgressBar: document.getElementById("hudProgressBar"),
            hudProgressText: document.getElementById("hudProgressText"),
            hudConditionInfo: document.getElementById("hudConditionInfo"),
            intervalCueText: document.getElementById("intervalCueText"),
            btnStart: document.getElementById("btnStart"),
            btnExitFullscreen: document.getElementById("btnExitFullscreen"),
            btnInterval1: document.getElementById("btnInterval1"),
            btnInterval2: document.getElementById("btnInterval2"),
            btnDownloadFullCSV: document.getElementById("btnDownloadFullCSV"),
            btnDownloadTrialCSV: document.getElementById("btnDownloadTrialCSV"),
            btnDownloadSummaryCSV: document.getElementById("btnDownloadSummaryCSV"),
            btnDownloadJSON: document.getElementById("btnDownloadJSON"),
            btnDownloadPlotPNG: document.getElementById("btnDownloadPlotPNG"),
            btnRestart: document.getElementById("btnRestart"),
            fitSummaryText: document.getElementById("fitSummaryText")
        };

        if (window.subjectManager) {
            window.subjectManager.bindUi({
                inputEl: "participantId",
                selectEl: "selectExistingSubject",
                btnGenEl: "btnGenSubjectCode",
                onSubjectChanged: (code) => {
                    this.participantId = code;
                }
            });
        } else {
            this.generateUniqueSubjectCode();
        }

        this._initHardwareDiagnostic();
        this._bindEvents();
        this._resizeCanvas();
        window.addEventListener("resize", () => this._resizeCanvas());
    }

    /**
     * Generate or load guaranteed unique Subject Code with sequential sensible naming
     * Format: OBS_01, OBS_02, OBS_03...
     */
    generateUniqueSubjectCode(forceNew = false) {
        let code;
        if (window.subjectManager) {
            code = forceNew 
                ? window.subjectManager.generateSensibleSubjectCode() 
                : window.subjectManager.getActiveSubject();
        } else {
            const urlParams = new URLSearchParams(window.location.search);
            const passedCode = urlParams.get("subject") || localStorage.getItem("vision_suite_subject_code");
            if (!forceNew && passedCode && passedCode.trim().length >= 2) {
                code = passedCode.trim().toUpperCase();
            } else {
                code = "OBS_01";
            }
        }
        localStorage.setItem("vision_suite_subject_code", code);

        const input = document.getElementById("participantId");
        if (input) {
            input.value = code;
            input.classList.remove("input-error");
        }
        this.participantId = code;
        const errEl = document.getElementById("subjectCodeError");
        if (errEl) errEl.style.display = "none";
        return code;
    }

    async _initHardwareDiagnostic() {
        if (!this.ui.hardwareReportContainer) return;

        this.ui.hardwareReportContainer.innerHTML = `
            <div class="diag-card" style="text-align: center; padding: 22px 16px;">
                <div style="color: var(--accent); font-weight: 700; margin-bottom: 6px; font-size: 14px;">
                    &#9881; Probing Display, GPU & Hardware Subsystems...
                </div>
                <div style="font-size: 12px; color: #94a3b8;">
                    Benchmarking refresh rate via requestAnimationFrame, probing 10-bit WebGL2 buffers, and detecting HDR OLED color pipeline...
                </div>
            </div>
        `;

        try {
            const diag = await this.hardwareChecker.runDiagnostic();
            this.hardwareChecker.renderReport(this.ui.hardwareReportContainer);
            this.exporter.setHardwareProfile(diag);

            const bitDepthSelect = document.getElementById("bitDepth");
            if (bitDepthSelect && (diag.display.isHDR || diag.assessment.isTenBitCapable)) {
                bitDepthSelect.value = "10";
            }
        } catch (e) {
            console.warn("Hardware diagnostic warning:", e);
        }
    }

    _resizeCanvas() {
        if (!this.canvas) return;
        const width = window.innerWidth;
        const height = window.innerHeight;
        const dpr = window.devicePixelRatio || 1;

        this.displayWidth = width;
        this.displayHeight = height;

        if (this.webglEngine && this.webglEngine.gl) {
            this.canvas.width = Math.round(width * dpr);
            this.canvas.height = Math.round(height * dpr);
            this.canvas.style.width = width + "px";
            this.canvas.style.height = height + "px";
            this.centerX = this.canvas.width / 2;
            this.centerY = this.canvas.height / 2;
            this.webglEngine.clearBackground();
        } else if (this.ctx) {
            this.canvas.width = width * dpr;
            this.canvas.height = height * dpr;
            this.ctx.resetTransform();
            this.ctx.scale(dpr, dpr);
            this.centerX = width / 2;
            this.centerY = height / 2;
            this._drawBackground();
        }
    }

    _drawBackground() {
        if (this.webglEngine && this.webglEngine.gl) {
            this.webglEngine.clearBackground();
        } else if (this.ctx) {
            this.ctx.fillStyle = "rgb(128, 128, 128)"; // Neutral mid-gray
            this.ctx.fillRect(0, 0, this.displayWidth, this.displayHeight);
        }
    }

    _initAudio() {
        if (!this.audioCtx) {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (AudioContext) {
                this.audioCtx = new AudioContext();
            }
        }
        if (this.audioCtx && this.audioCtx.state === "suspended") {
            this.audioCtx.resume();
        }
    }

    /**
     * Play tone for interval cue or feedback
     * @param {number} freq - Pitch in Hz
     * @param {number} durationMs - Duration in ms
     * @param {string} [type='sine']
     */
    _playTone(freq, durationMs, type = "sine") {
        if (!this.audioCtx) return;
        try {
            const now = this.audioCtx.currentTime;
            const osc = this.audioCtx.createOscillator();
            const gain = this.audioCtx.createGain();

            osc.type = type;
            osc.frequency.setValueAtTime(freq, now);

            gain.gain.setValueAtTime(0.001, now);
            gain.gain.linearRampToValueAtTime(0.12, now + 0.01);
            gain.gain.exponentialRampToValueAtTime(0.001, now + durationMs / 1000);

            osc.connect(gain);
            gain.connect(this.audioCtx.destination);

            osc.start(now);
            osc.stop(now + (durationMs + 5) / 1000);
        } catch (e) {
            // Audio error gracefully handled
        }
    }

    _bindEvents() {
        if (this.ui.btnStart) {
            this.ui.btnStart.addEventListener("click", () => this.startExperiment());
        }

        if (this.ui.btnExitFullscreen) {
            this.ui.btnExitFullscreen.addEventListener("click", () => {
                this.exitFullscreen();
                this.showSetupScreen();
            });
        }

        // Keyboard controls for 2IFC:
        // '1', 'ArrowLeft', 'F' -> Interval 1
        // '2', 'ArrowRight', 'J' -> Interval 2
        window.addEventListener("keydown", (e) => {
            if (this.state !== "AWAIT_RESPONSE") return;
            const key = e.key.toLowerCase();
            if (key === "1" || key === "arrowleft" || key === "f") {
                e.preventDefault();
                this.handleResponse(1);
            } else if (key === "2" || key === "arrowright" || key === "j") {
                e.preventDefault();
                this.handleResponse(2);
            } else if (key === "escape") {
                this.exitFullscreen();
                this.showSetupScreen();
            }
        });

        // Touch / Click on-screen buttons
        if (this.ui.btnInterval1) {
            this.ui.btnInterval1.addEventListener("click", () => {
                if (this.state === "AWAIT_RESPONSE") this.handleResponse(1);
            });
        }
        if (this.ui.btnInterval2) {
            this.ui.btnInterval2.addEventListener("click", () => {
                if (this.state === "AWAIT_RESPONSE") this.handleResponse(2);
            });
        }

        // Subject code controls
        if (this.ui.btnGenSubjectCode) {
            this.ui.btnGenSubjectCode.addEventListener("click", () => {
                this.generateUniqueSubjectCode(true);
            });
        }

        const pInput = document.getElementById("participantId");
        if (pInput) {
            pInput.addEventListener("input", () => {
                pInput.classList.remove("input-error");
                const err = document.getElementById("subjectCodeError");
                if (err) err.style.display = "none";
            });
        }

        // Export controls
        if (this.ui.btnDownloadFullCSV) {
            this.ui.btnDownloadFullCSV.addEventListener("click", () => {
                const summary = this.getCSFSummary();
                const modelFit = this.exporter.fitCSFModel(summary.map(s => ({
                    spatialFreq: s.spatialFreqCpd,
                    sensitivity: s.sensitivity
                })));
                const fullCsv = this.exporter.generateFullCSF_CSV(summary, modelFit);
                const timestampStr = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
                this.exporter.downloadBlob(fullCsv, `csf_data_${this.participantId}_${timestampStr}.csv`, "text/csv");
            });
        }

        if (this.ui.btnDownloadTrialCSV) {
            this.ui.btnDownloadTrialCSV.addEventListener("click", () => {
                const csv = this.exporter.generateTrialCSV();
                this.exporter.downloadBlob(csv, `csf_2ifc_trials_${this.participantId}.csv`, "text/csv");
            });
        }

        if (this.ui.btnDownloadSummaryCSV) {
            this.ui.btnDownloadSummaryCSV.addEventListener("click", () => {
                const summary = this.getCSFSummary();
                const modelFit = this.exporter.fitCSFModel(summary.map(s => ({
                    spatialFreq: s.spatialFreqCpd,
                    sensitivity: s.sensitivity
                })));
                const csv = this.exporter.generateSummaryCSV(summary, modelFit);
                this.exporter.downloadBlob(csv, `csf_2ifc_summary_${this.participantId}.csv`, "text/csv");
            });
        }

        if (this.ui.btnDownloadJSON) {
            this.ui.btnDownloadJSON.addEventListener("click", () => {
                const summary = this.getCSFSummary();
                const modelFit = this.exporter.fitCSFModel(summary.map(s => ({
                    spatialFreq: s.spatialFreqCpd,
                    sensitivity: s.sensitivity
                })));
                const json = this.exporter.generateJSON(summary, modelFit);
                this.exporter.downloadBlob(json, `csf_2ifc_data_${this.participantId}.json`, "application/json");
            });
        }

        if (this.ui.btnDownloadPlotPNG) {
            this.ui.btnDownloadPlotPNG.addEventListener("click", () => {
                if (!this.plotCanvas) return;
                const link = document.createElement("a");
                link.download = `csf_2ifc_plot_${this.participantId}.png`;
                link.href = this.plotCanvas.toDataURL("image/png");
                link.click();
            });
        }

        if (this.ui.btnRestart) {
            this.ui.btnRestart.addEventListener("click", () => this.showSetupScreen());
        }
    }

    readConfigFromUI() {
        const participantInput = document.getElementById("participantId");
        if (participantInput && participantInput.value.trim()) {
            this.participantId = participantInput.value.trim();
        }

        const sfRangeSelect = document.getElementById("sfRange");
        if (sfRangeSelect) {
            if (sfRangeSelect.value === "ultrawide") {
                this.spatialFrequencies = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
            } else if (sfRangeSelect.value === "extended") {
                this.spatialFrequencies = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
            } else if (sfRangeSelect.value === "standard") {
                this.spatialFrequencies = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
            }
        }

        const trialsSelect = document.getElementById("trialsPerCondition");
        if (trialsSelect) {
            this.trialsPerCondition = parseInt(trialsSelect.value, 10) || 20;
        }

        const envelopeSelect = document.getElementById("envelopeSize");
        if (envelopeSelect) {
            this.sigmaDeg = parseFloat(envelopeSelect.value) || 0.5;
        }

        const distanceInput = document.getElementById("viewingDistance");
        if (distanceInput) {
            this.viewingDistanceCm = parseFloat(distanceInput.value) || 57.3;
        }

        const dpiSelect = document.getElementById("displayDpi");
        if (dpiSelect) {
            const dpi = parseFloat(dpiSelect.value) || 96.0;
            this.pixelsPerCm = dpi / 2.54;
        }

        const bitDepthSelect = document.getElementById("bitDepth");
        if (bitDepthSelect) {
            this.bitDepth = parseInt(bitDepthSelect.value, 10) || 8;
        }

        const ditherCheck = document.getElementById("enableDithering");
        if (ditherCheck) {
            this.enableDithering = ditherCheck.checked;
        }

        this.renderer = new GaborRenderer({
            viewingDistanceCm: this.viewingDistanceCm,
            pixelsPerCm: this.pixelsPerCm,
            bgLuminance: 128,
            sigmaDeg: this.sigmaDeg, // Constant envelope size
            bitDepth: this.bitDepth,
            enableDithering: this.enableDithering
        });

        this.exporter.setMetadata({
            participantId: this.participantId,
            viewingDistanceCm: this.viewingDistanceCm,
            pixelsPerDegree: this.renderer.pixelsPerDegree,
            bitDepth: this.bitDepth,
            ditheringEnabled: this.enableDithering,
            sigmaDeg: this.sigmaDeg,
            spatialFrequencies: this.spatialFrequencies,
            trialsPerCondition: this.trialsPerCondition
        });
    }

    requestFullscreen() {
        const el = document.documentElement;
        if (el.requestFullscreen) {
            el.requestFullscreen().catch(() => {});
        } else if (el.webkitRequestFullscreen) {
            el.webkitRequestFullscreen().catch(() => {});
        }
    }

    exitFullscreen() {
        if (document.fullscreenElement) {
            if (document.exitFullscreen) document.exitFullscreen().catch(() => {});
            else if (document.webkitExitFullscreen) document.webkitExitFullscreen().catch(() => {});
        }
    }

    startExperiment() {
        // Validate Unique Subject Code
        const participantInput = document.getElementById("participantId");
        const code = participantInput ? participantInput.value.trim() : "";
        if (!code || code.length < 2) {
            const errEl = document.getElementById("subjectCodeError");
            if (errEl) {
                errEl.textContent = "Please enter or auto-generate a unique Subject Code before starting the experiment.";
                errEl.style.display = "block";
            }
            if (participantInput) {
                participantInput.classList.add("input-error");
                participantInput.focus();
            }
            return;
        }

        this.participantId = code;
        this._initAudio();
        this.readConfigFromUI();

        // Attach latest hardware profile to exporter
        if (this.hardwareChecker.results) {
            this.exporter.setHardwareProfile(this.hardwareChecker.results);
        }

        this.requestFullscreen();

        document.body.classList.add("experiment-active");
        this.ui.screenSetup.style.display = "none";
        this.ui.screenExperiment.classList.add("active");
        this.ui.screenResults.style.display = "none";

        this._resizeCanvas();

        // Initialize independent QUEST+ instances for each spatial frequency
        // For 2IFC 2AFC, guessRate = 0.5 (strict 50% chance level)
        this.questInstances.clear();
        this.spatialFrequencies.forEach(sf => {
            const q = new QuestPlus({
                thresholds: this._linspace(-3.0, 0.0, 61),
                intensities: this._linspace(-2.9, 0.0, 51),
                slope: 2.5,
                guessRate: 0.5, // 2AFC chance level
                lapseRate: 0.02
            });
            this.questInstances.set(sf, q);
        });

        // Build randomized interleaved trial queue
        // Balanced mini-blocks: each block has 1 trial of each of the 6 SFs
        this.trialQueue = [];
        for (let block = 0; block < this.trialsPerCondition; block++) {
            const blockSFs = [...this.spatialFrequencies];
            this._shuffle(blockSFs);
            blockSFs.forEach(sf => {
                this.trialQueue.push({
                    spatialFreqCpd: sf,
                    blockIndex: block
                });
            });
        }

        this.currentTrialIndex = 0;
        this.updateProgressUI();
        this.scheduleNextTrial();
    }

    _linspace(start, end, n) {
        const step = (end - start) / (n - 1);
        const arr = [];
        for (let i = 0; i < n; i++) arr.push(start + i * step);
        return arr;
    }

    _shuffle(arr) {
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
    }

    showSetupScreen() {
        this.state = "IDLE";
        if (this.timerId) clearTimeout(this.timerId);
        this.exitFullscreen();
        document.body.classList.remove("experiment-active");
        this.ui.screenSetup.style.display = "block";
        this.ui.screenExperiment.classList.remove("active");
        this.ui.screenResults.style.display = "none";
    }

    showResultsScreen() {
        this.state = "COMPLETED";
        if (this.timerId) clearTimeout(this.timerId);
        this.exitFullscreen();
        document.body.classList.remove("experiment-active");
        this.ui.screenSetup.style.display = "none";
        this.ui.screenExperiment.classList.remove("active");
        this.ui.screenResults.style.display = "block";

        if (this.hardwareChecker.results) {
            this.exporter.setHardwareProfile(this.hardwareChecker.results);
        }

        this.renderFinalCSFPlot();

        // Automatically save full CSV dataset
        try {
            const summary = this.getCSFSummary();
            const modelFit = this.exporter.fitCSFModel(summary.map(s => ({
                spatialFreq: s.spatialFreqCpd,
                sensitivity: s.sensitivity
            })));
            const fullCsv = this.exporter.generateFullCSF_CSV(summary, modelFit);
            const timestampStr = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
            const filename = `csf_data_${this.participantId}_${timestampStr}.csv`;

            this.exporter.downloadBlob(fullCsv, filename, "text/csv");

            // Save session payload to localStorage for unified results viewer
            const sessionPayload = {
                subject: this.participantId,
                date: new Date().toISOString(),
                summary: summary,
                modelFit: modelFit,
                hardware: this.hardwareChecker.results
            };
            try {
                localStorage.setItem("vision_suite_csf_last", JSON.stringify(sessionPayload));
                localStorage.setItem("vision_suite_csf_" + this.participantId, JSON.stringify(sessionPayload));
            } catch (storageErr) {
                console.warn("Could not save to localStorage:", storageErr);
            }

            const banner = document.getElementById("csvAutoDownloadBanner");
            const bannerText = document.getElementById("csvAutoDownloadText");
            if (banner && bannerText) {
                bannerText.textContent = `Experiment CSV automatically saved as "${filename}"`;
                banner.style.display = "flex";
            }
        } catch (e) {
            console.error("Auto CSV export failed:", e);
        }
    }

    scheduleNextTrial() {
        if (this.currentTrialIndex >= this.trialQueue.length) {
            this.showResultsScreen();
            return;
        }

        const trialInfo = this.trialQueue[this.currentTrialIndex];
        const quest = this.questInstances.get(trialInfo.spatialFreqCpd);

        const optimalLogContrast = quest.getOptimalIntensity();
        const testContrast = Math.pow(10, optimalLogContrast);

        // In 2IFC, target randomly appears in Interval 1 or Interval 2 with p = 0.5
        const targetInterval = Math.random() < 0.5 ? 1 : 2;
        const randomPhase = Math.random() * 2 * Math.PI;

        this.currentTrialData = {
            spatialFreqCpd: trialInfo.spatialFreqCpd,
            sigmaDeg: this.sigmaDeg,
            logContrast: optimalLogContrast,
            contrast: testContrast,
            phase: randomPhase,
            targetInterval: targetInterval
        };

        // Transition to FIXATION_PRE
        this.state = "FIXATION_PRE";
        this._enableResponseControls(false);
        this._drawStimulusDisplay(null);

        this.timerId = setTimeout(() => {
            this.presentInterval1();
        }, this.preFixationDuration);
    }

    /**
     * Compute constant half-width for the spatial bounding box across all tested spatial frequencies
     * @returns {number} box half-size in pixels
     */
    getStimulusBoxSize() {
        if (!this.renderer) return 100;
        const maxEffSigma = Math.max(...this.spatialFrequencies.map(sf => this.renderer.getEffectiveSigmaDeg(sf)));
        const maxPatchRadius = Math.max(28, Math.ceil(3.5 * this.renderer.degToPixels(maxEffSigma)));
        const maxScreenHalf = Math.floor(Math.min(this.displayWidth, this.displayHeight) * 0.40);
        return Math.min(maxScreenHalf, maxPatchRadius + 14);
    }

    /**
     * Renders the stimulus display:
     * 1. Fullscreen neutral mid-gray background (rgb(128, 128, 128))
     * 2. Stimulus is rendered strictly inside the spatial bounding box (clipped)
     * 3. Spatial bounding box outline (2px crisp dark-gray border)
     * 4. Four outer fixation crosshairs positioned strictly OUTSIDE the bounding box.
     *    Crosshairs never enter the box and NEVER overlap the stimulus!
     * @param {HTMLCanvasElement|null} [gaborCanvas=null]
     */
    _drawStimulusDisplay(gaborCanvas = null, feedback = null) {
        const boxHalf = this.getStimulusBoxSize();

        // WebGL2 Hardware Render Path
        if (this.webglEngine && this.webglEngine.gl) {
            if (gaborCanvas && this.currentTrialData) {
                const sfPixels = this.currentTrialData.spatialFreqCpd / this.renderer.pixelsPerDegree;
                const effSigma = this.renderer.getEffectiveSigmaDeg(this.currentTrialData.spatialFreqCpd);
                const sigmaPixels = this.renderer.degToPixels(effSigma);
                this.webglEngine.renderGabor({
                    centerX: this.centerX,
                    centerY: this.centerY,
                    spatialFreqPixels: sfPixels,
                    contrast: this.currentTrialData.contrast,
                    orientationRad: 0,
                    phaseRad: this.currentTrialData.phase,
                    sigmaPixels: sigmaPixels,
                    boxHalfSizePx: boxHalf,
                    feedback: feedback
                });
            } else {
                this.webglEngine.renderFixationOnly({
                    centerX: this.centerX,
                    centerY: this.centerY,
                    boxHalfSizePx: boxHalf,
                    feedback: feedback
                });
            }
            return;
        }

        // Canvas 2D Fallback
        this._drawBackground();

        const x = this.centerX - boxHalf;
        const y = this.centerY - boxHalf;
        const size = 2 * boxHalf;

        // Feedback colors: green for correct, red for incorrect, dark-gray for neutral
        let boxColor = "rgba(40, 40, 40, 0.85)";
        let armColor = "rgba(25, 25, 25, 0.95)";
        if (feedback === "CORRECT") {
            boxColor = "rgba(46, 125, 50, 0.95)";
            armColor = "#2e7d32";
        } else if (feedback === "INCORRECT") {
            boxColor = "rgba(198, 40, 40, 0.95)";
            armColor = "#c62828";
        }

        // 1. If Gabor patch is present, clip and draw strictly INSIDE the bounding box
        if (gaborCanvas) {
            this.ctx.save();
            this.ctx.beginPath();
            this.ctx.rect(x, y, size, size);
            this.ctx.clip();
            this.renderer.drawCentered(this.ctx, this.centerX, this.centerY, gaborCanvas);
            this.ctx.restore();
        }

        // 2. Draw spatial bounding box border
        this.ctx.strokeStyle = boxColor;
        this.ctx.lineWidth = feedback ? 2.5 : 2;
        this.ctx.strokeRect(x, y, size, size);

        // 3. Draw outer fixation crosshairs strictly OUTSIDE the bounding box.
        const armLen = 18;
        const gap = 6;

        this.ctx.strokeStyle = armColor;
        this.ctx.lineWidth = 2.5;

        // Top arm
        this.ctx.beginPath();
        this.ctx.moveTo(this.centerX, y - gap - armLen);
        this.ctx.lineTo(this.centerX, y - gap);
        this.ctx.stroke();

        // Bottom arm
        this.ctx.beginPath();
        this.ctx.moveTo(this.centerX, y + size + gap);
        this.ctx.lineTo(this.centerX, y + size + gap + armLen);
        this.ctx.stroke();

        // Left arm
        this.ctx.beginPath();
        this.ctx.moveTo(x - gap - armLen, this.centerY);
        this.ctx.lineTo(x - gap, this.centerY);
        this.ctx.stroke();

        // Right arm
        this.ctx.beginPath();
        this.ctx.moveTo(x + size + gap, this.centerY);
        this.ctx.lineTo(x + size + gap + armLen, this.centerY);
        this.ctx.stroke();
    }

    _drawFixationMarker() {
        this._drawStimulusDisplay(null);
    }

    _setIntervalCueText(txt) {
        // Disabled: Visual numbers (1, 2) and question mark (?) removed
        if (this.ui.intervalCueText) {
            this.ui.intervalCueText.textContent = "";
        }
    }

    /**
     * Presentation of Interval 1 (VSync-locked 120 ms)
     * Preceded by Beep 1 (500 Hz, 80 ms)
     */
    presentInterval1() {
        this.state = "INTERVAL_1";
        this._playTone(500, 80, "sine"); // Beep 1 @ 500 Hz

        const isTarget = (this.currentTrialData.targetInterval === 1);
        const boxHalf = this.getStimulusBoxSize();
        const sfPixels = this.currentTrialData.spatialFreqCpd / this.renderer.pixelsPerDegree;
        const effSigma = this.renderer.getEffectiveSigmaDeg(this.currentTrialData.spatialFreqCpd);
        const sigmaPixels = this.renderer.degToPixels(effSigma);

        // WebGL Hardware Presentation Path (VSync-Locked)
        if (this.webglEngine && this.webglEngine.gl) {
            this.webglEngine.presentInterval({
                type: isTarget ? "GABOR" : "BLANK",
                durationMs: this.intervalDuration,
                params: {
                    centerX: this.centerX,
                    centerY: this.centerY,
                    spatialFreqPixels: sfPixels,
                    contrast: isTarget ? this.currentTrialData.contrast : 0.0,
                    orientationRad: 0,
                    phaseRad: this.currentTrialData.phase,
                    sigmaPixels: sigmaPixels,
                    boxHalfSizePx: boxHalf,
                    feedback: null
                },
                onComplete: (telemetry) => {
                    this.currentTrialData.telemetry1 = telemetry;
                    this.presentISI();
                }
            });
            return;
        }

        // Fallback Canvas 2D
        if (isTarget) {
            const gaborCanvas = this.renderer.createGaborCanvas({
                spatialFreqCpd: this.currentTrialData.spatialFreqCpd,
                contrast: this.currentTrialData.contrast,
                orientationDeg: 0,
                phase: this.currentTrialData.phase,
                sigmaDeg: this.sigmaDeg
            });
            this._drawStimulusDisplay(gaborCanvas);
        } else {
            this._drawStimulusDisplay(null);
        }

        this.timerId = setTimeout(() => {
            this.presentISI();
        }, this.intervalDuration);
    }

    /**
     * Inter-Stimulus Interval (ISI) between Interval 1 and 2 (400 ms)
     */
    presentISI() {
        this.state = "ISI";
        this._drawStimulusDisplay(null);

        this.timerId = setTimeout(() => {
            this.presentInterval2();
        }, this.interStimulusInterval);
    }

    /**
     * Presentation of Interval 2 (VSync-locked 120 ms)
     * Preceded by Beep 2 (750 Hz, 80 ms)
     */
    presentInterval2() {
        this.state = "INTERVAL_2";
        this._playTone(750, 80, "sine"); // Beep 2 @ 750 Hz

        const isTarget = (this.currentTrialData.targetInterval === 2);
        const boxHalf = this.getStimulusBoxSize();
        const sfPixels = this.currentTrialData.spatialFreqCpd / this.renderer.pixelsPerDegree;
        const effSigma = this.renderer.getEffectiveSigmaDeg(this.currentTrialData.spatialFreqCpd);
        const sigmaPixels = this.renderer.degToPixels(effSigma);

        // WebGL Hardware Presentation Path (VSync-Locked)
        if (this.webglEngine && this.webglEngine.gl) {
            this.webglEngine.presentInterval({
                type: isTarget ? "GABOR" : "BLANK",
                durationMs: this.intervalDuration,
                params: {
                    centerX: this.centerX,
                    centerY: this.centerY,
                    spatialFreqPixels: sfPixels,
                    contrast: isTarget ? this.currentTrialData.contrast : 0.0,
                    orientationRad: 0,
                    phaseRad: this.currentTrialData.phase,
                    sigmaPixels: sigmaPixels,
                    boxHalfSizePx: boxHalf,
                    feedback: null
                },
                onComplete: (telemetry) => {
                    this.currentTrialData.telemetry2 = telemetry;
                    this.awaitObserverResponse();
                }
            });
            return;
        }

        // Fallback Canvas 2D
        if (isTarget) {
            const gaborCanvas = this.renderer.createGaborCanvas({
                spatialFreqCpd: this.currentTrialData.spatialFreqCpd,
                contrast: this.currentTrialData.contrast,
                orientationDeg: 0,
                phase: this.currentTrialData.phase,
                sigmaDeg: this.sigmaDeg
            });
            this._drawStimulusDisplay(gaborCanvas);
        } else {
            this._drawStimulusDisplay(null);
        }

        this.timerId = setTimeout(() => {
            this.awaitObserverResponse();
        }, this.intervalDuration);
    }

    awaitObserverResponse() {
        this.state = "AWAIT_RESPONSE";
        this._drawStimulusDisplay(null);
        this._enableResponseControls(true);
        this.responseWindowStartTime = performance.now();
    }

    _enableResponseControls(enabled) {
        if (this.ui.btnInterval1) this.ui.btnInterval1.disabled = !enabled;
        if (this.ui.btnInterval2) this.ui.btnInterval2.disabled = !enabled;
    }

    /**
     * Process 2IFC Response: Observer selects Interval 1 or Interval 2
     * @param {number} chosenInterval - 1 or 2
     */
    handleResponse(chosenInterval) {
        if (this.state !== "AWAIT_RESPONSE") return;
        this._enableResponseControls(false);

        const rtMs = performance.now() - this.responseWindowStartTime;
        const targetInterval = this.currentTrialData.targetInterval;
        const isCorrect = (chosenInterval === targetInterval);

        // Feedback Tone
        if (isCorrect) {
            this._playTone(880, 80, "sine"); // Pleasant high chime
        } else {
            this._playTone(220, 120, "triangle"); // Low buzz
        }

        // Update QUEST+ for this spatial frequency condition
        const quest = this.questInstances.get(this.currentTrialData.spatialFreqCpd);
        quest.update(this.currentTrialData.logContrast, isCorrect ? 1 : 0);

        const estThreshold = quest.getThresholdMean();
        const estStd = quest.getThresholdStd();
        const estLinearThreshold = Math.pow(10, estThreshold);
        const estSensitivity = 1.0 / Math.max(1e-5, estLinearThreshold);

        // Log trial
        this.exporter.logTrial({
            spatialFreqCpd: this.currentTrialData.spatialFreqCpd,
            sigmaDeg: this.sigmaDeg,
            targetInterval: targetInterval,
            responseInterval: chosenInterval,
            isCorrect: isCorrect,
            reactionTimeMs: rtMs,
            testLog10Contrast: this.currentTrialData.logContrast,
            testContrast: this.currentTrialData.contrast,
            phaseRad: this.currentTrialData.phase,
            phaseDeg: this.currentTrialData.phase * (180.0 / Math.PI),
            estimatedThresholdMean: estThreshold,
            estimatedThresholdStd: estStd,
            estimatedContrastThreshold: estLinearThreshold,
            estimatedSensitivity: estSensitivity,
            entropy: quest.getCurrentEntropy(),
            durationMsTarget: this.intervalDuration,
            actualDurationMs1: this.currentTrialData.telemetry1 ? this.currentTrialData.telemetry1.actualDurationMs : this.intervalDuration,
            actualFrames1: this.currentTrialData.telemetry1 ? this.currentTrialData.telemetry1.actualFrames : null,
            jitterMs1: this.currentTrialData.telemetry1 ? this.currentTrialData.telemetry1.jitterMs : null,
            droppedFrames1: this.currentTrialData.telemetry1 ? this.currentTrialData.telemetry1.droppedFrames : 0,
            actualDurationMs2: this.currentTrialData.telemetry2 ? this.currentTrialData.telemetry2.actualDurationMs : this.intervalDuration,
            actualFrames2: this.currentTrialData.telemetry2 ? this.currentTrialData.telemetry2.actualFrames : null,
            jitterMs2: this.currentTrialData.telemetry2 ? this.currentTrialData.telemetry2.jitterMs : null,
            droppedFrames2: this.currentTrialData.telemetry2 ? this.currentTrialData.telemetry2.droppedFrames : 0
        });

        this.currentTrialIndex++;
        this.updateProgressUI();

        // Inter-Trial Interval with immediate visual feedback (green/red outline)
        this.state = "ITI";
        this._drawStimulusDisplay(null, isCorrect ? "CORRECT" : "INCORRECT");

        this.timerId = setTimeout(() => {
            this.scheduleNextTrial();
        }, this.interTrialInterval);
    }

    updateProgressUI() {
        const total = this.trialQueue.length;
        const current = this.currentTrialIndex;
        const pct = Math.round((current / total) * 100);

        if (this.ui.hudProgressBar) {
            this.ui.hudProgressBar.style.width = `${pct}%`;
        }
        if (this.ui.hudProgressText) {
            this.ui.hudProgressText.textContent = `Trial ${current} / ${total} (${pct}%)`;
        }
        if (this.ui.hudConditionInfo) {
            const acc = this.exporter.getAccuracyStats();
            this.ui.hudConditionInfo.textContent = `2IFC Accuracy: ${acc.accuracyPct.toFixed(1)}% | Envelope: ${this.sigmaDeg.toFixed(2)}°`;
        }
    }

    getCSFSummary() {
        return this.spatialFrequencies.map(sf => {
            const q = this.questInstances.get(sf);
            const meanLogT = q.getThresholdMean();
            const stdLogT = q.getThresholdStd();
            const linT = Math.pow(10, meanLogT);
            const sens = 1.0 / Math.max(1e-5, linT);
            return {
                spatialFreqCpd: sf,
                sigmaDeg: this.sigmaDeg,
                thresholdLog10: meanLogT,
                thresholdStd: stdLogT,
                contrastThreshold: linT,
                sensitivity: sens,
                trialsCount: q.history.length
            };
        });
    }

    /**
     * Render the high-resolution double-logarithmic Contrast Sensitivity Function Plot
     */
    renderFinalCSFPlot() {
        if (!this.plotCanvas) return;
        const summary = this.getCSFSummary();
        const acc = this.exporter.getAccuracyStats();

        // Fit parabolic log-sensitivity model
        const modelFit = this.exporter.fitCSFModel(summary.map(s => ({
            spatialFreq: s.spatialFreqCpd,
            sensitivity: s.sensitivity
        })));

        // Update textual summary card
        if (this.ui.fitSummaryText) {
            let fitHtml = `
                <div class="fit-grid">
                    <div class="fit-item">
                        <span class="fit-label">Subject Code:</span>
                        <span class="fit-value text-accent">${this.participantId}</span>
                    </div>
                    <div class="fit-item">
                        <span class="fit-label">Peak Sensitivity ($S_{\\max}$):</span>
                        <span class="fit-value">${modelFit.fitted ? modelFit.sMax.toFixed(1) : "--"} (C = ${(100/modelFit.sMax).toFixed(2)}%)</span>
                    </div>
                    <div class="fit-item">
                        <span class="fit-label">Peak Spatial Frequency ($f_{\\max}$):</span>
                        <span class="fit-value">${modelFit.fitted ? modelFit.fMax.toFixed(2) + " cpd" : "--"}</span>
                    </div>
                    <div class="fit-item">
                        <span class="fit-label">Curve Bandwidth ($k$):</span>
                        <span class="fit-value">${modelFit.fitted ? modelFit.bandwidthK.toFixed(3) : "--"}</span>
                    </div>
                    <div class="fit-item">
                        <span class="fit-label">Model Fit ($R^2$):</span>
                        <span class="fit-value">${modelFit.fitted ? (modelFit.rSquared * 100).toFixed(1) + "%" : "--"}</span>
                    </div>
                    <div class="fit-item">
                        <span class="fit-label">Overall 2AFC Accuracy:</span>
                        <span class="fit-value">${acc.accuracyPct.toFixed(1)}% (${acc.nCorrect}/${acc.totalTrials})</span>
                    </div>
                    <div class="fit-item">
                        <span class="fit-label">Task Paradigm:</span>
                        <span class="fit-value">Two-Interval Forced Choice (2IFC)</span>
                    </div>
                    <div class="fit-item">
                        <span class="fit-label">Constant Envelope ($\\sigma$):</span>
                        <span class="fit-value">${this.sigmaDeg.toFixed(2)}&deg; visual angle</span>
                    </div>
                    <div class="fit-item">
                        <span class="fit-label">Display Mode:</span>
                        <span class="fit-value">${this.bitDepth}-Bit (Dithering: ${this.enableDithering ? 'ON' : 'OFF'})</span>
                    </div>
                </div>
            `;
            this.ui.fitSummaryText.innerHTML = fitHtml;
        }

        // Setup high-DPI canvas
        const dpr = window.devicePixelRatio || 1;
        const cssWidth = 720;
        const cssHeight = 460;
        this.plotCanvas.width = cssWidth * dpr;
        this.plotCanvas.height = cssHeight * dpr;
        this.plotCanvas.style.width = `${cssWidth}px`;
        this.plotCanvas.style.height = `${cssHeight}px`;

        const ctx = this.plotCanvas.getContext("2d");
        ctx.resetTransform();
        ctx.scale(dpr, dpr);

        // Clear background
        ctx.fillStyle = "#1e1e24";
        ctx.fillRect(0, 0, cssWidth, cssHeight);

        // Plot margins
        const pad = { left: 75, right: 35, top: 40, bottom: 65 };
        const plotW = cssWidth - pad.left - pad.right;
        const plotH = cssHeight - pad.top - pad.bottom;

        // Log scales:
        // Dynamically span full range of tested spatial frequencies (e.g. 0.125 to 32 cpd)
        const minSF = Math.min(...this.spatialFrequencies);
        const maxSF = Math.max(...this.spatialFrequencies);
        const logXMin = Math.log10(minSF * 0.7);
        const logXMax = Math.log10(maxSF * 1.35);
        // Y-axis: 1.0 to 1000.0 sensitivity (0.1% to 100% contrast)
        const logYMin = Math.log10(1.0);
        const logYMax = Math.log10(1000.0);

        const xToPx = (f) => pad.left + ((Math.log10(f) - logXMin) / (logXMax - logXMin)) * plotW;
        const yToPx = (s) => pad.top + plotH - ((Math.log10(s) - logYMin) / (logYMax - logYMin)) * plotH;

        // Draw Plot Area Background
        ctx.fillStyle = "#131318";
        ctx.fillRect(pad.left, pad.top, plotW, plotH);

        // Highlight Sub-LSB / Dithered Domain (Sensitivity > 128)
        const floor8bitSens = 128.0; // 1 / 0.0078125
        const floorY = yToPx(floor8bitSens);

        ctx.fillStyle = "rgba(79, 140, 255, 0.08)";
        ctx.fillRect(pad.left, pad.top, plotW, floorY - pad.top);

        ctx.fillStyle = "rgba(120, 180, 255, 0.4)";
        ctx.font = "italic 11px system-ui, sans-serif";
        ctx.fillText("Pseudo-10-Bit Dithered Domain (Sensitivity > 128, Contrast < 0.78%)", pad.left + 12, pad.top + 20);

        // Draw 8-Bit Hardware Quantization Floor Line
        ctx.strokeStyle = "rgba(255, 100, 100, 0.75)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(pad.left, floorY);
        ctx.lineTo(pad.left + plotW, floorY);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = "rgba(255, 120, 120, 0.9)";
        ctx.font = "bold 11px system-ui, sans-serif";
        ctx.fillText("Physical 8-Bit Display Floor (S = 128, C = 0.78%)", pad.left + plotW - 275, floorY - 6);

        // Draw Grid Lines & Ticks: X-axis (Spatial Frequencies)
        const xTicks = this.spatialFrequencies;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
        ctx.lineWidth = 1;
        ctx.textAlign = "center";
        ctx.fillStyle = "#a0a0b0";
        ctx.font = "11px system-ui, sans-serif";

        xTicks.forEach(f => {
            const px = xToPx(f);
            ctx.beginPath();
            ctx.moveTo(px, pad.top);
            ctx.lineTo(px, pad.top + plotH);
            ctx.stroke();

            const label = f < 1 ? (f === 0.125 ? "0.12" : f.toString()) : f.toString();
            ctx.fillText(label, px, pad.top + plotH + 18);
        });

        // X-axis label
        ctx.fillStyle = "#ffffff";
        ctx.font = "13px system-ui, sans-serif";
        ctx.fillText("Spatial Frequency (cycles per degree)", pad.left + plotW / 2, pad.top + plotH + 44);

        // Draw Grid Lines & Ticks: Y-axis (Contrast Sensitivity)
        const yTicks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];
        const yLabels = [1, 10, 100, 1000];

        yTicks.forEach(s => {
            const py = yToPx(s);
            const isMajor = yLabels.includes(s);
            ctx.strokeStyle = isMajor ? "rgba(255, 255, 255, 0.15)" : "rgba(255, 255, 255, 0.05)";
            ctx.beginPath();
            ctx.moveTo(pad.left, py);
            ctx.lineTo(pad.left + plotW, py);
            ctx.stroke();

            if (isMajor) {
                ctx.textAlign = "right";
                ctx.fillStyle = "#a0a0b0";
                ctx.font = "12px system-ui, sans-serif";
                ctx.fillText(s.toString(), pad.left - 10, py + 4);
            }
        });

        // Y-axis label
        ctx.save();
        ctx.translate(pad.left - 48, pad.top + plotH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "center";
        ctx.fillStyle = "#ffffff";
        ctx.font = "13px system-ui, sans-serif";
        ctx.fillText("Contrast Sensitivity (1 / Threshold Contrast)", 0, 0);
        ctx.restore();

        // Draw Plot Border
        ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
        ctx.lineWidth = 1;
        ctx.strokeRect(pad.left, pad.top, plotW, plotH);

        // Draw Smooth Fitted Model Curve (Parabolic Log-Sensitivity)
        if (modelFit && modelFit.fitted) {
            ctx.save();
            ctx.beginPath();
            ctx.rect(pad.left, pad.top, plotW, plotH);
            ctx.clip();

            ctx.strokeStyle = "#40c4ff";
            ctx.lineWidth = 3;
            ctx.beginPath();

            const nCurveSteps = 100;
            for (let i = 0; i <= nCurveSteps; i++) {
                const logF = logXMin + (i / nCurveSteps) * (logXMax - logXMin);
                const f = Math.pow(10, logF);
                const s = modelFit.predict(f);
                const px = xToPx(f);
                const py = yToPx(s);

                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }
            ctx.stroke();
            ctx.restore();
        }

        // Draw Empirical Data Points & Error Bars
        summary.forEach(d => {
            const px = xToPx(d.spatialFreqCpd);
            const py = yToPx(d.sensitivity);

            const sensHigh = 1.0 / Math.pow(10, d.thresholdLog10 - d.thresholdStd);
            const sensLow = 1.0 / Math.pow(10, d.thresholdLog10 + d.thresholdStd);
            const pyHigh = yToPx(sensHigh);
            const pyLow = yToPx(sensLow);

            ctx.strokeStyle = "#ffeb3b";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(px, pyLow);
            ctx.lineTo(px, pyHigh);
            ctx.stroke();

            const capW = 5;
            ctx.beginPath();
            ctx.moveTo(px - capW, pyLow);
            ctx.lineTo(px + capW, pyLow);
            ctx.moveTo(px - capW, pyHigh);
            ctx.lineTo(px + capW, pyHigh);
            ctx.stroke();

            ctx.fillStyle = d.sensitivity > floor8bitSens ? "#00e676" : "#ffeb3b";
            ctx.strokeStyle = "#131318";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(px, py, 6, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
        });

        // Plot Title
        ctx.textAlign = "left";
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 15px system-ui, sans-serif";
        ctx.fillText(`Contrast Sensitivity Function (2IFC 2AFC, Envelope σ = ${this.sigmaDeg.toFixed(2)}°)`, pad.left, pad.top - 15);

        // Legend
        const legX = pad.left + plotW - 210;
        const legY = pad.top + 36;
        ctx.fillStyle = "rgba(20, 20, 30, 0.85)";
        ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
        ctx.lineWidth = 1;
        ctx.fillRect(legX, legY, 200, 75);
        ctx.strokeRect(legX, legY, 200, 75);

        ctx.fillStyle = "#ffeb3b";
        ctx.beginPath();
        ctx.arc(legX + 16, legY + 20, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = "#e0e0e0";
        ctx.font = "11px system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("Empirical (Mean ± 1 SD)", legX + 30, legY + 24);

        ctx.strokeStyle = "#40c4ff";
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(legX + 8, legY + 44);
        ctx.lineTo(legX + 24, legY + 44);
        ctx.stroke();
        ctx.fillStyle = "#e0e0e0";
        ctx.fillText("Parabolic CSF Model Fit", legX + 30, legY + 48);

        ctx.strokeStyle = "rgba(255, 100, 100, 0.9)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(legX + 8, legY + 64);
        ctx.lineTo(legX + 24, legY + 64);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#e0e0e0";
        ctx.fillText("8-Bit Undithered Floor", legX + 30, legY + 67);
    }
}

window.addEventListener("DOMContentLoaded", () => {
    window.csfExp = new CSFExperiment();
});
