/**
 * experiment.js - Main Vision Science Experiment Controller
 * Orchestrates 12 interleaved QUEST+ conditions (3 sizes x 4 eccentricities)
 * for ON vs. OFF Difference of Gaussians (DoG) discrimination.
 */

class VisionExperiment {
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

        // Independent variables specification
        this.sizesDeg = [0.15, 0.30, 0.60]; // 3 DoG spatial scales
        this.eccentricitiesDeg = [2.0, 4.0, 6.0, 8.0]; // 4 retinal distances from fixation
        this.trialsPerCondition = 20; // 20 trials per QUEST+ condition (configurable)

        // Instantiate modules
        this.renderer = new DogRenderer();
        this.exporter = new DataExporter();

        // 12 experimental conditions
        this.conditions = [];
        this._initConditions();

        // Experiment state
        this.state = "INSTRUCTIONS"; // INSTRUCTIONS, FIXATION, STIMULUS, RESPONSE, FINISHED
        this.currentCondition = null;
        this.currentPolarity = null; // "ON" or "OFF"
        this.currentContrast = null;
        this.currentEccentricity = null;
        this.currentAngleRad = 0;
        this.stimulusPatchCanvas = null;

        this.stimulusDurationMs = 120; // 120 ms flash
        this.fixationDurationMs = 600;
        this.trialStartTime = 0;
        this.stimulusOnsetTime = 0;
        this.totalTrialsCompleted = 0;
        this.totalTrialsExpected = this.conditions.length * this.trialsPerCondition;

        // Web Audio API feedback
        this.audioCtx = null;

        this._setupCanvasSize();
        this._setupListeners();
        this._initSubjectCode();
    }

    _initSubjectCode() {
        if (typeof window !== "undefined" && window.subjectManager) {
            window.subjectManager.bindUi({
                inputEl: "participantId",
                selectEl: "selectExistingSubject",
                btnGenEl: "btnGenSubjectCode",
                onSubjectChanged: (code) => {
                    this.exporter.setParticipantId(code);
                }
            });
            const activeCode = window.subjectManager.getActiveSubject();
            this.exporter.setParticipantId(activeCode);
        } else {
            const urlParams = new URLSearchParams(window.location.search);
            const passedCode = urlParams.get("subject") || localStorage.getItem("vision_suite_subject_code") || "OBS_01";
            const input = document.getElementById("participantId");
            if (input) input.value = passedCode;
            this.exporter.setParticipantId(passedCode);
        }
    }

    _initConditions() {
        let idCounter = 1;
        for (const size of this.sizesDeg) {
            for (const ecc of this.eccentricitiesDeg) {
                this.conditions.push({
                    id: idCounter++,
                    sizeDeg: size,
                    eccDeg: ecc,
                    name: `Size_${size}deg_Ecc_${ecc}deg`,
                    quest: new QuestPlus({
                        thresholds: this._linspace(-2.5, 0.0, 51),
                        intensities: this._linspace(-2.4, 0.0, 41),
                        slope: 2.5,
                        guessRate: 0.5,
                        lapseRate: 0.02
                    }),
                    trialsCompleted: 0
                });
            }
        }
    }

    _linspace(start, end, n) {
        const step = (end - start) / (n - 1);
        const arr = [];
        for (let i = 0; i < n; i++) arr.push(start + i * step);
        return arr;
    }

    _setupCanvasSize() {
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = Math.round(window.innerWidth * dpr);
        this.canvas.height = Math.round(window.innerHeight * dpr);
        this.canvas.style.width = window.innerWidth + "px";
        this.canvas.style.height = window.innerHeight + "px";
        if (this.ctx) {
            this.ctx.resetTransform();
            this.ctx.scale(dpr, dpr);
        }
        window.addEventListener("resize", () => {
            this.canvas.width = Math.round(window.innerWidth * dpr);
            this.canvas.height = Math.round(window.innerHeight * dpr);
            this.canvas.style.width = window.innerWidth + "px";
            this.canvas.style.height = window.innerHeight + "px";
            if (this.ctx) {
                this.ctx.resetTransform();
                this.ctx.scale(dpr, dpr);
            }
            if (this.state === "FIXATION" || this.state === "RESPONSE") {
                this.drawFixation();
            }
        });
    }

    _setupListeners() {
        // Keyboard responses
        window.addEventListener("keydown", (e) => {
            if (this.state === "RESPONSE") {
                if (e.key === "ArrowUp" || e.key === "f" || e.key === "F") {
                    this.handleResponse("ON");
                } else if (e.key === "ArrowDown" || e.key === "j" || e.key === "J") {
                    this.handleResponse("OFF");
                }
            } else if (this.state === "INSTRUCTIONS" && (e.key === " " || e.key === "Enter")) {
                this.start();
            }
        });

        // Start button listener
        document.getElementById("btnStart")?.addEventListener("click", () => {
            if (this.state === "INSTRUCTIONS") this.start();
        });

        // Export button listeners
        document.getElementById("btnDownloadTrials")?.addEventListener("click", () => {
            this.exporter.downloadTrialDataCsv();
        });
        document.getElementById("btnDownloadSummary")?.addEventListener("click", () => {
            this.exporter.downloadSummaryCsv(this.conditions);
        });
        document.getElementById("btnDownloadJson")?.addEventListener("click", () => {
            this.exporter.downloadFullSessionJson(this.conditions);
        });
    }

    setTrialsPerCondition(count) {
        this.trialsPerCondition = count;
        this.totalTrialsExpected = this.conditions.length * this.trialsPerCondition;
        this._updateProgress();
    }

    _initAudio() {
        if (!this.audioCtx) {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (AudioContextClass) this.audioCtx = new AudioContextClass();
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

    start() {
        this._initAudio();
        const modal = document.getElementById("instructionsModal");
        if (modal) modal.style.display = "none";
        this.totalTrialsCompleted = 0;
        this.nextTrial();
    }

    getActiveConditions() {
        return this.conditions.filter(c => c.quest.history.length < this.trialsPerCondition);
    }

    nextTrial() {
        const active = this.getActiveConditions();
        if (active.length === 0) {
            this.finishExperiment();
            return;
        }

        // Randomly pick from remaining active conditions (QUEST+ condition interleaving)
        const randIdx = Math.floor(Math.random() * active.length);
        this.currentCondition = active[randIdx];

        // Optimal test intensity recommended by QUEST+ (Watson, 2017)
        const logContrast = this.currentCondition.quest.getOptimalIntensity();
        this.currentContrast = Math.max(0.005, Math.min(1.0, Math.pow(10, logContrast)));

        // Randomize polarity (50% ON, 50% OFF)
        this.currentPolarity = Math.random() < 0.5 ? "ON" : "OFF";

        // Randomize presentation side (left or right horizontal meridian)
        this.currentAngleRad = Math.random() < 0.5 ? 0 : Math.PI;

        // Pre-render stimulus patch
        this.stimulusPatchCanvas = this.renderer.createDogCanvas({
            sigmaCenterDeg: this.currentCondition.sizeDeg,
            contrast: this.currentContrast,
            polarity: this.currentPolarity
        });

        // Enter fixation state
        this.state = "FIXATION";
        this.drawFixation();

        setTimeout(() => {
            this.presentStimulus();
        }, this.fixationDurationMs);
    }

    drawFixation(feedbackColor = null) {
        const cx = this.canvas.width / 2;
        const cy = this.canvas.height / 2;

        if (this.webglEngine && this.webglEngine.gl) {
            let fb = null;
            if (feedbackColor === "#00ff00" || feedbackColor === "rgba(0, 255, 0, 0.8)" || feedbackColor === "#4caf50") fb = "CORRECT";
            else if (feedbackColor === "#ff0000" || feedbackColor === "rgba(255, 0, 0, 0.8)" || feedbackColor === "#f44336") fb = "INCORRECT";

            this.webglEngine.renderFixationOnly({
                centerX: cx,
                centerY: cy,
                boxHalfSizePx: 50,
                feedback: fb,
                showBoundingBox: false,
                fixationCenter: [cx, cy]
            });
            return;
        }

        // 2D Canvas Fallback: Single stationary central fixation cross, NO bounding box
        this.ctx.fillStyle = `rgb(${this.renderer.bgLuminance}, ${this.renderer.bgLuminance}, ${this.renderer.bgLuminance})`;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Fixation cross (green/red feedback or neutral dark-gray)
        this.ctx.strokeStyle = feedbackColor || "#222222";
        this.ctx.lineWidth = feedbackColor ? 3.5 : 2.5;
        const arm = 10;
        this.ctx.beginPath();
        this.ctx.moveTo(cx - arm, cy);
        this.ctx.lineTo(cx + arm, cy);
        this.ctx.moveTo(cx, cy - arm);
        this.ctx.lineTo(cx, cy + arm);
        this.ctx.stroke();
    }

    presentStimulus() {
        this.state = "STIMULUS";

        const cx = this.canvas.width / 2;
        const cy = this.canvas.height / 2;
        const eccPx = this.renderer.degToPixels(this.currentCondition.eccDeg);
        const stimX = Math.round(cx + eccPx * Math.cos(this.currentAngleRad));
        const stimY = Math.round(cy + eccPx * Math.sin(this.currentAngleRad));
        const sigmaCPx = this.renderer.degToPixels(this.currentCondition.sizeDeg);
        const sigmaSPx = 2.0 * sigmaCPx;
        const boxHalfPx = Math.ceil(3.5 * sigmaSPx) + 12;

        if (this.webglEngine && this.webglEngine.gl) {
            this.webglEngine.presentInterval({
                type: "DOG",
                durationMs: this.stimulusDurationMs,
                params: {
                    centerX: stimX,
                    centerY: stimY,
                    sigmaCPixels: sigmaCPx,
                    sigmaSPixels: sigmaSPx,
                    pedestalContrast: 0.0,
                    incrementContrast: this.currentContrast,
                    polarity: this.currentPolarity,
                    boxHalfSizePx: boxHalfPx,
                    feedback: null,
                    showBoundingBox: false,
                    fixationCenter: [cx, cy]
                },
                onComplete: (telemetry) => {
                    this.state = "RESPONSE";
                    this.drawFixation();
                }
            });
            this.stimulusOnsetTime = performance.now();
            return;
        }

        // 2D Canvas Fallback: Draw single central fixation, then draw stimulus at eccentricity without bounding box
        this.drawFixation();
        this.renderer.drawAtEccentricity(
            this.ctx,
            cx,
            cy,
            this.currentCondition.eccDeg,
            this.currentAngleRad,
            this.stimulusPatchCanvas
        );
        this.stimulusOnsetTime = performance.now();

        setTimeout(() => {
            this.state = "RESPONSE";
            this.drawFixation();
        }, this.stimulusDurationMs);
    }

    handleResponse(participantChoice) {
        if (this.state !== "RESPONSE") return;

        this._initAudio();
        const rt = performance.now() - this.stimulusOnsetTime;
        const isCorrect = participantChoice === this.currentPolarity;

        // Auditory feedback: High chime (correct) vs Low buzz (incorrect)
        if (isCorrect) {
            this.playTone(880, 80, "sine", 0.22); // High pleasant chime
        } else {
            this.playTone(220, 140, "triangle", 0.20); // Low error buzz
        }

        // Update this condition's QUEST+ model
        const logIntensityTested = Math.log10(this.currentContrast);
        this.currentCondition.quest.update(logIntensityTested, isCorrect);
        this.currentCondition.trialsCompleted++;
        this.totalTrialsCompleted++;

        // Log trial record
        this.exporter.logTrial({
            trialIndex: this.totalTrialsCompleted,
            conditionId: this.currentCondition.id,
            sizeCenterDeg: this.currentCondition.sizeDeg,
            eccentricityDeg: this.currentCondition.eccDeg,
            polarity: this.currentPolarity,
            testedContrast: Number(this.currentContrast.toFixed(5)),
            testedLogIntensity: Number(logIntensityTested.toFixed(4)),
            participantResponse: participantChoice,
            correct: isCorrect ? 1 : 0,
            reactionTimeMs: Math.round(rt),
            estimatedThresholdLog: Number(this.currentCondition.quest.getThresholdMean().toFixed(4)),
            estimatedThresholdLinear: Number(this.currentCondition.quest.getLinearContrastThreshold().toFixed(5)),
            posteriorEntropyBits: Number(this.currentCondition.quest.getCurrentEntropy().toFixed(3))
        });

        this._updateProgress();

        // Inter-trial interval (ITI) of 450 ms with visual feedback (green/red fixation cross)
        this.state = "ITI";
        const feedbackColor = isCorrect ? "#4caf50" : "#f44336";
        this.drawFixation(feedbackColor);
        setTimeout(() => {
            this.nextTrial();
        }, 450);
    }

    _updateProgress() {
        const pct = Math.round((this.totalTrialsCompleted / this.totalTrialsExpected) * 100);
        const progressBar = document.getElementById("progressBar");
        const progressText = document.getElementById("progressText");
        if (progressBar) progressBar.style.width = `${pct}%`;
        if (progressText) progressText.innerText = `Trial ${this.totalTrialsCompleted} / ${this.totalTrialsExpected} (${pct}%)`;
    }

    finishExperiment() {
        this.state = "FINISHED";
        const touchControls = document.getElementById("touchControls");
        if (touchControls) touchControls.style.display = "none";
        document.getElementById("resultsModal").style.display = "block";
        this._renderSummaryTable();
        this._renderPsychometricPlot();
    }

    _renderSummaryTable() {
        const tbody = document.getElementById("summaryTableBody");
        if (!tbody) return;
        tbody.innerHTML = "";

        const summary = this.exporter.generateConditionSummary(this.conditions);
        for (const row of summary) {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>${row.conditionId}</td>
                <td>${row.sizeCenterDeg}°</td>
                <td>${row.eccentricityDeg}°</td>
                <td>${row.trialsCompleted}</td>
                <td><strong>${row.linearContrastPercent}%</strong></td>
                <td>${row.contrastSensitivity}</td>
                <td>±${row.thresholdStdDev}</td>
            `;
            tbody.appendChild(tr);
        }
    }

    _renderPsychometricPlot() {
        const plotCanvas = document.getElementById("plotCanvas");
        if (!plotCanvas) return;
        const ctx = plotCanvas.getContext("2d");
        const w = plotCanvas.width;
        const h = plotCanvas.height;

        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, w, h);

        const padLeft = 60, padRight = 30, padTop = 40, padBottom = 50;
        const plotW = w - padLeft - padRight;
        const plotH = h - padTop - padBottom;

        // Axes
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(padLeft, padTop);
        ctx.lineTo(padLeft, padTop + plotH);
        ctx.lineTo(padLeft + plotW, padTop + plotH);
        ctx.stroke();

        // Labels
        ctx.fillStyle = "#111";
        ctx.font = "13px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Retinal Eccentricity (degrees from fixation)", padLeft + plotW / 2, h - 15);

        ctx.save();
        ctx.translate(18, padTop + plotH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText("Contrast Threshold (%)", 0, 0);
        ctx.restore();

        const colors = ["#e63946", "#2a9d8f", "#457b9d"];
        const eccs = this.eccentricitiesDeg;
        const minEcc = 0, maxEcc = 10;
        const minThresh = 0, maxThresh = 50;

        // Grid lines
        ctx.strokeStyle = "#eee";
        for (let t = 10; t <= 50; t += 10) {
            const y = padTop + plotH - (t / maxThresh) * plotH;
            ctx.beginPath();
            ctx.moveTo(padLeft, y);
            ctx.lineTo(padLeft + plotW, y);
            ctx.stroke();
            ctx.fillStyle = "#666";
            ctx.textAlign = "right";
            ctx.fillText(`${t}%`, padLeft - 8, y + 4);
        }

        // Plot curves for each size
        this.sizesDeg.forEach((size, sIdx) => {
            const sizeConds = this.conditions.filter(c => c.sizeDeg === size);
            ctx.strokeStyle = colors[sIdx];
            ctx.fillStyle = colors[sIdx];
            ctx.lineWidth = 2.5;

            ctx.beginPath();
            sizeConds.forEach((c, idx) => {
                const threshPct = c.quest.getLinearContrastThreshold() * 100.0;
                const x = padLeft + (c.eccDeg / maxEcc) * plotW;
                const y = padTop + plotH - (Math.min(maxThresh, threshPct) / maxThresh) * plotH;
                if (idx === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();

            // Draw points
            sizeConds.forEach((c) => {
                const threshPct = c.quest.getLinearContrastThreshold() * 100.0;
                const x = padLeft + (c.eccDeg / maxEcc) * plotW;
                const y = padTop + plotH - (Math.min(maxThresh, threshPct) / maxThresh) * plotH;
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, 2 * Math.PI);
                ctx.fill();
            });

            // Legend
            const legX = padLeft + 20 + sIdx * 110;
            const legY = padTop - 15;
            ctx.fillStyle = colors[sIdx];
            ctx.fillRect(legX, legY - 8, 12, 12);
            ctx.fillStyle = "#333";
            ctx.textAlign = "left";
            ctx.fillText(`σ=${size}°`, legX + 18, legY + 2);
        });
    }
}

// Auto-instantiate upon DOM ready
window.addEventListener("DOMContentLoaded", () => {
    window.experiment = new VisionExperiment();
});
