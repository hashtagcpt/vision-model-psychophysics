/**
 * screen_calibrator.js - Physical Screen & Viewing Distance Calibration Module
 * 
 * Implements:
 * 1. Physical Screen Metric Calibration via Standard ID-1 Credit Card (ISO/IEC 7810: 85.60 mm x 53.98 mm)
 * 2. Visual Angle Calculation (Pixels per visual degree at any viewing distance d)
 * 3. Optional Virtual Chinrest / Blind Spot Distance Measurement (Li et al., 2020, Scientific Reports)
 * 4. Cross-experiment persistent storage via localStorage ('vision_suite_screen_calibration')
 */

class ScreenCalibrator {
    static CARD_WIDTH_MM = 85.60;
    static CARD_HEIGHT_MM = 53.98;
    static CARD_ASPECT_RATIO = 85.60 / 53.98; // ~1.58577
    static BLIND_SPOT_ECCENTRICITY_DEG = 15.5; // Average human optic disc temporal center

    // Standard display presets
    static PRESETS = {
        surface_pro: { name: "Surface Pro 11 / 9 / 8 (267 PPI)", ppi: 267, defaultDistCm: 57.3 },
        macbook_retina: { name: "MacBook Pro Retina (227-254 PPI)", ppi: 227, defaultDistCm: 57.3 },
        fhd_laptop: { name: "Standard 15.6\" FHD Laptop (141 PPI)", ppi: 141, defaultDistCm: 57.3 },
        fhd_desktop: { name: "Standard 24\" FHD Desktop Monitor (92 PPI)", ppi: 92, defaultDistCm: 65.0 },
        uhd_desktop: { name: "Standard 27\" 4K Desktop Monitor (163 PPI)", ppi: 163, defaultDistCm: 65.0 }
    };

    constructor(options = {}) {
        this.onCalibrateCallback = options.onCalibrated || null;
        this.currentCardWidthPx = 320; // Default pixel width
        this.viewingDistanceCm = 57.3; // Default distance where 1 cm = 1 deg
        this.modalElement = null;
        this.blindSpotActive = false;
        this.blindSpotAnimId = null;

        this.loadStoredCalibration();
    }

    /**
     * Load existing calibration from localStorage or initialize with sensible defaults
     */
    loadStoredCalibration() {
        if (typeof localStorage !== "undefined") {
            try {
                const raw = localStorage.getItem("vision_suite_screen_calibration");
                if (raw) {
                    const parsed = JSON.parse(raw);
                    this.currentCardWidthPx = parsed.cardWidthPx || 320;
                    this.viewingDistanceCm = parsed.viewingDistanceCm || 57.3;
                    this.pxPerMm = parsed.pxPerMm;
                    this.pxPerCm = parsed.pxPerCm;
                    this.dpi = parsed.dpi;
                    this.pxPerDeg = parsed.pxPerDeg;
                    this.calibrationMethod = parsed.method || "credit_card";
                    this.calibratedAt = parsed.calibratedAt;
                    return parsed;
                }
            } catch (e) {
                console.warn("Could not load stored calibration:", e);
            }
        }

        // Default initialization based on Surface Pro OLED or DPR
        const dpr = (typeof window !== "undefined" && window.devicePixelRatio) ? window.devicePixelRatio : 2.0;
        const estimatedPpi = dpr >= 2.0 ? 267 : 141; // Surface Pro vs Laptop
        this.pxPerMm = (estimatedPpi / 25.4);
        this.currentCardWidthPx = Math.round(ScreenCalibrator.CARD_WIDTH_MM * this.pxPerMm);
        this.recomputeMetrics();
        return this.getCalibrationData();
    }

    /**
     * Recompute derived metrics: pixels per mm, DPI, pixels per degree
     */
    recomputeMetrics() {
        this.pxPerMm = this.currentCardWidthPx / ScreenCalibrator.CARD_WIDTH_MM;
        this.pxPerCm = this.pxPerMm * 10.0;
        this.dpi = Math.round(this.pxPerMm * 25.4 * 10) / 10;

        // Visual angle geometry:
        // Size in cm for theta = 1 deg at distance d cm: s = 2 * d * tan(0.5 deg)
        const rad1Deg = (1.0 * Math.PI) / 180.0;
        const cmPerDeg = 2.0 * this.viewingDistanceCm * Math.tan(rad1Deg / 2.0);
        this.pxPerDeg = Math.round(cmPerDeg * this.pxPerCm * 100) / 100;
        this.degPerPx = 1.0 / this.pxPerDeg;
    }

    /**
     * Save calibration to localStorage
     */
    saveCalibration(method = "credit_card") {
        this.recomputeMetrics();
        const data = {
            cardWidthPx: Math.round(this.currentCardWidthPx),
            viewingDistanceCm: Number(this.viewingDistanceCm.toFixed(1)),
            pxPerMm: Number(this.pxPerMm.toFixed(3)),
            pxPerCm: Number(this.pxPerCm.toFixed(2)),
            dpi: this.dpi,
            pxPerDeg: this.pxPerDeg,
            degPerPx: Number(this.degPerPx.toFixed(5)),
            method: method,
            calibratedAt: new Date().toISOString(),
            dpr: (typeof window !== "undefined" && window.devicePixelRatio) ? window.devicePixelRatio : 1,
            screenCssResolution: (typeof window !== "undefined" && window.screen) ? `${window.screen.width}x${window.screen.height}` : "Unknown",
            screenPhysicalResolution: (typeof window !== "undefined" && window.screen) ? `${Math.round(window.screen.width * (window.devicePixelRatio || 1))}x${Math.round(window.screen.height * (window.devicePixelRatio || 1))}` : "Unknown"
        };

        if (typeof localStorage !== "undefined") {
            try {
                localStorage.setItem("vision_suite_screen_calibration", JSON.stringify(data));
            } catch (e) {
                console.warn("Could not save to localStorage:", e);
            }
        }

        if (typeof this.onCalibrateCallback === "function") {
            this.onCalibrateCallback(data);
        }
        return data;
    }

    /**
     * Static accessor for any experiment to retrieve active calibration
     */
    static getCalibration() {
        if (typeof localStorage !== "undefined") {
            try {
                const raw = localStorage.getItem("vision_suite_screen_calibration");
                if (raw) return JSON.parse(raw);
            } catch (e) {}
        }

        // Fallback default
        const dpr = (typeof window !== "undefined" && window.devicePixelRatio) ? window.devicePixelRatio : 2.0;
        const ppi = dpr >= 2.0 ? 267 : 141;
        const pxPerMm = ppi / 25.4;
        const pxPerCm = pxPerMm * 10;
        const dist = 57.3;
        const rad1Deg = (1.0 * Math.PI) / 180.0;
        const cmPerDeg = 2.0 * dist * Math.tan(rad1Deg / 2.0);
        return {
            cardWidthPx: Math.round(ScreenCalibrator.CARD_WIDTH_MM * pxPerMm),
            viewingDistanceCm: dist,
            pxPerMm: pxPerMm,
            pxPerCm: pxPerCm,
            dpi: ppi,
            pxPerDeg: cmPerDeg * pxPerCm,
            degPerPx: 1.0 / (cmPerDeg * pxPerCm),
            method: "default_preset",
            calibratedAt: null
        };
    }

    getCalibrationData() {
        return {
            cardWidthPx: Math.round(this.currentCardWidthPx),
            viewingDistanceCm: this.viewingDistanceCm,
            pxPerMm: this.pxPerMm,
            pxPerCm: this.pxPerCm,
            dpi: this.dpi,
            pxPerDeg: this.pxPerDeg,
            degPerPx: this.degPerPx
        };
    }

    /**
     * Open interactive calibration modal
     */
    openCalibrationModal() {
        if (this.modalElement) {
            this.modalElement.style.display = "flex";
            this.updateModalUI();
            return;
        }

        this._createModalDOM();
        this.updateModalUI();
    }

    closeModal() {
        if (this.modalElement) {
            this.modalElement.style.display = "none";
        }
        if (this.blindSpotActive) {
            this.stopBlindSpotTest();
        }
    }

    _createModalDOM() {
        const modal = document.createElement("div");
        modal.id = "screenCalibratorModal";
        modal.className = "calib-modal-overlay";
        modal.innerHTML = `
            <div class="calib-modal-card">
                <div class="calib-header">
                    <div>
                        <h2>Display &amp; Viewing Distance Calibration</h2>
                        <p class="calib-subtitle">
                            Ensure exact physical millimeters on screen and exact visual angles (&deg;) for psychophysics.
                        </p>
                    </div>
                    <button class="calib-btn-close" id="btnCalibClose">&times;</button>
                </div>

                <!-- Tabs: Card Calibration vs. Blind Spot -->
                <div class="calib-tabs">
                    <button class="calib-tab-btn active" id="tabCardBtn">💳 Standard Credit Card Sizing</button>
                    <button class="calib-tab-btn" id="tabBlindSpotBtn">👁 Virtual Chinrest (Blind Spot Distance)</button>
                    <button class="calib-tab-btn" id="tabPresetsBtn">🖥 Display Presets</button>
                </div>

                <!-- Tab 1: Credit Card Calibration -->
                <div class="calib-tab-content active" id="calibCardSection">
                    <p class="calib-instruction">
                        1. Place any physical credit card, ID card, or driver's license directly against your screen.<br>
                        2. Adjust the slider or buttons below until the on-screen card <strong>precisely matches</strong> your real card.
                    </p>

                    <!-- Interactive Card Preview -->
                    <div class="calib-card-stage">
                        <div class="calib-credit-card" id="calibCardPreview">
                            <div class="card-chip"></div>
                            <div class="card-contactless">)))</div>
                            <div class="card-number-lines">
                                <span>••••</span> <span>••••</span> <span>••••</span> <span>••••</span>
                            </div>
                            <div class="card-footer-info">
                                <span class="card-holder">STANDARD ID-1 (85.60 &times; 53.98 mm)</span>
                                <span class="card-brand">VISION SUITE</span>
                            </div>
                        </div>
                    </div>

                    <!-- Card Size Slider Controls -->
                    <div class="calib-control-row">
                        <button class="calib-btn-step" id="btnCardMinus5">-5px</button>
                        <button class="calib-btn-step" id="btnCardMinus1">-1px</button>
                        <input type="range" id="sliderCardWidth" min="180" max="600" step="1" value="${Math.round(this.currentCardWidthPx)}">
                        <button class="calib-btn-step" id="btnCardPlus1">+1px</button>
                        <button class="calib-btn-step" id="btnCardPlus5">+5px</button>
                    </div>

                    <!-- Viewing Distance Input -->
                    <div class="calib-dist-section">
                        <label for="inputViewingDist"><strong>Viewing Distance:</strong></label>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <input type="number" id="inputViewingDist" value="${this.viewingDistanceCm}" step="0.5" min="20" max="150" class="calib-dist-input">
                            <span>cm</span>
                            <span class="calib-dist-hint">(57.3 cm ensures exactly 1.0 cm on screen = 1.0&deg; visual angle)</span>
                        </div>
                    </div>
                </div>

                <!-- Tab 2: Virtual Chinrest (Blind Spot Test) -->
                <div class="calib-tab-content" id="calibBlindSpotSection" style="display: none;">
                    <p class="calib-instruction">
                        <strong>Measure viewing distance without a ruler</strong> using your physiological blind spot (Li et al., 2020):<br>
                        1. Close or cover your <strong>LEFT eye</strong>.<br>
                        2. Stare steadily at the <strong>red fixation cross (+)</strong> with your RIGHT eye.<br>
                        3. The black dot will sweep horizontally. Click <strong>"Disappeared"</strong> the moment the dot vanishes into your blind spot.
                    </p>
                    <div class="blindspot-stage" id="blindSpotStage">
                        <div class="blindspot-cross" id="blindSpotCross">+</div>
                        <div class="blindspot-dot" id="blindSpotDot"></div>
                    </div>
                    <div class="blindspot-actions">
                        <button class="calib-btn-action" id="btnStartBlindSpot">▶ Start Blind Spot Sweep</button>
                        <button class="calib-btn-action" id="btnMarkBlindSpot" style="display: none; background: #e53935; color: #fff;">Dot Disappeared! [Spacebar]</button>
                        <span id="blindSpotResultText" style="font-size: 13px; color: #00e676; font-weight: bold;"></span>
                    </div>
                </div>

                <!-- Tab 3: Display Presets -->
                <div class="calib-tab-content" id="calibPresetsSection" style="display: none;">
                    <p class="calib-instruction">
                        Select a verified hardware profile to auto-fill display metrics:
                    </p>
                    <div class="preset-grid">
                        <button class="preset-btn" onclick="window.screenCalibrator.applyPreset('surface_pro')">
                            <strong>Surface Pro 11 / 9 / 8 (OLED)</strong><br>
                            <span>2880 &times; 1920 &bull; 267 PPI &bull; 105 px/cm</span>
                        </button>
                        <button class="preset-btn" onclick="window.screenCalibrator.applyPreset('macbook_retina')">
                            <strong>MacBook Pro Retina</strong><br>
                            <span>227 - 254 PPI &bull; 89 - 100 px/cm</span>
                        </button>
                        <button class="preset-btn" onclick="window.screenCalibrator.applyPreset('fhd_laptop')">
                            <strong>Standard FHD 15.6" Laptop</strong><br>
                            <span>1920 &times; 1080 &bull; 141 PPI &bull; 55.5 px/cm</span>
                        </button>
                        <button class="preset-btn" onclick="window.screenCalibrator.applyPreset('fhd_desktop')">
                            <strong>Standard 24" Desktop Monitor</strong><br>
                            <span>1920 &times; 1080 &bull; 92 PPI &bull; 36.2 px/cm</span>
                        </button>
                        <button class="preset-btn" onclick="window.screenCalibrator.applyPreset('uhd_desktop')">
                            <strong>Standard 27" 4K Desktop Monitor</strong><br>
                            <span>3840 &times; 2160 &bull; 163 PPI &bull; 64.2 px/cm</span>
                        </button>
                    </div>
                </div>

                <!-- Real-Time Metrics Output -->
                <div class="calib-metrics-summary">
                    <div class="metric-item">
                        <span class="m-label">Display DPI</span>
                        <span class="m-val" id="metricDpi">${this.dpi} PPI</span>
                    </div>
                    <div class="metric-item">
                        <span class="m-label">Pixels per cm</span>
                        <span class="m-val" id="metricPxCm">${this.pxPerCm ? this.pxPerCm.toFixed(1) : 0} px/cm</span>
                    </div>
                    <div class="metric-item">
                        <span class="m-label">Distance</span>
                        <span class="m-val" id="metricDist">${this.viewingDistanceCm} cm</span>
                    </div>
                    <div class="metric-item highlight">
                        <span class="m-label">Pixels per Degree</span>
                        <span class="m-val" id="metricPxDeg"><strong>${this.pxPerDeg ? this.pxPerDeg.toFixed(1) : 0} px/&deg;</strong></span>
                    </div>
                </div>

                <!-- Footer Actions -->
                <div class="calib-footer">
                    <button class="calib-btn-secondary" id="btnResetCalib">↺ Reset Defaults</button>
                    <div style="display: flex; gap: 10px;">
                        <button class="calib-btn-secondary" id="btnCancelCalib">Cancel</button>
                        <button class="calib-btn-primary" id="btnSaveCalib">✓ Save &amp; Apply Calibration</button>
                    </div>
                </div>
            </div>
        `;

        // Inject Styles
        this._injectModalStyles();

        document.body.appendChild(modal);
        this.modalElement = modal;

        this._bindModalEvents();
    }

    _injectModalStyles() {
        if (document.getElementById("screenCalibratorStyles")) return;
        const style = document.createElement("style");
        style.id = "screenCalibratorStyles";
        style.textContent = `
            .calib-modal-overlay {
                position: fixed;
                top: 0; left: 0; right: 0; bottom: 0;
                background: rgba(8, 8, 12, 0.88);
                backdrop-filter: blur(10px);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 9999;
                padding: 16px;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }
            .calib-modal-card {
                background: #15151e;
                border: 1px solid #2d2d42;
                border-radius: 16px;
                padding: 26px 30px;
                max-width: 680px;
                width: 100%;
                max-height: 94vh;
                overflow-y: auto;
                box-shadow: 0 16px 48px rgba(0,0,0,0.7);
                color: #e2e8f0;
            }
            .calib-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 14px;
            }
            .calib-header h2 {
                margin: 0;
                font-size: 20px;
                font-weight: 700;
                color: #fff;
            }
            .calib-subtitle {
                font-size: 13px;
                color: #94a3b8;
                margin: 4px 0 0 0;
            }
            .calib-btn-close {
                background: transparent;
                border: none;
                color: #94a3b8;
                font-size: 24px;
                cursor: pointer;
                line-height: 1;
            }
            .calib-btn-close:hover { color: #fff; }
            .calib-tabs {
                display: flex;
                gap: 8px;
                margin-bottom: 18px;
                border-bottom: 1px solid #28283a;
                padding-bottom: 8px;
            }
            .calib-tab-btn {
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.1);
                color: #94a3b8;
                font-size: 12px;
                font-weight: 600;
                padding: 7px 14px;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.15s ease;
            }
            .calib-tab-btn.active {
                background: #00bcd4;
                color: #000;
                border-color: #00bcd4;
            }
            .calib-instruction {
                font-size: 13px;
                color: #cbd5e1;
                line-height: 1.5;
                margin-bottom: 14px;
            }
            .calib-card-stage {
                background: #0d0d14;
                border: 1px dashed #3a3a55;
                border-radius: 12px;
                padding: 30px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 18px;
                min-height: 220px;
            }
            .calib-credit-card {
                background: linear-gradient(135deg, #1e293b, #0f172a);
                border: 2px solid rgba(0, 188, 212, 0.6);
                border-radius: 10px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.6);
                position: relative;
                padding: 16px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                box-sizing: border-box;
                user-select: none;
                transition: width 0.05s ease, height 0.05s ease;
            }
            .card-chip {
                width: 32px;
                height: 24px;
                background: linear-gradient(135deg, #f59e0b, #d97706);
                border-radius: 4px;
                border: 1px solid #b45309;
            }
            .card-contactless {
                position: absolute;
                top: 14px;
                right: 18px;
                color: #94a3b8;
                font-weight: bold;
                font-size: 14px;
            }
            .card-number-lines {
                display: flex;
                gap: 10px;
                font-family: monospace;
                font-size: 12px;
                color: #64748b;
                letter-spacing: 2px;
            }
            .card-footer-info {
                display: flex;
                justify-content: space-between;
                font-size: 9px;
                font-weight: 700;
                color: #94a3b8;
                letter-spacing: 0.5px;
            }
            .calib-control-row {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 16px;
            }
            .calib-control-row input[type="range"] {
                flex: 1;
                accent-color: #00bcd4;
                height: 6px;
                cursor: pointer;
            }
            .calib-btn-step {
                background: #252538;
                border: 1px solid #3d3d5c;
                color: #e2e8f0;
                padding: 6px 10px;
                font-size: 11px;
                font-weight: bold;
                border-radius: 6px;
                cursor: pointer;
            }
            .calib-btn-step:hover { background: #35354e; }
            .calib-dist-section {
                display: flex;
                align-items: center;
                gap: 12px;
                background: rgba(255,255,255,0.03);
                padding: 10px 14px;
                border-radius: 8px;
                border: 1px solid #28283a;
                margin-bottom: 18px;
            }
            .calib-dist-input {
                width: 70px;
                background: #0b0b12;
                border: 1px solid #3d3d5c;
                border-radius: 6px;
                color: #fff;
                padding: 6px 8px;
                font-size: 14px;
                font-weight: bold;
            }
            .calib-dist-hint {
                font-size: 11px;
                color: #94a3b8;
            }
            .calib-metrics-summary {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 10px;
                background: #0f0f18;
                border: 1px solid #28283c;
                border-radius: 10px;
                padding: 12px 14px;
                margin-bottom: 20px;
                text-align: center;
            }
            .metric-item .m-label {
                display: block;
                font-size: 10px;
                color: #94a3b8;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 4px;
            }
            .metric-item .m-val {
                font-size: 14px;
                font-weight: 700;
                color: #f1f5f9;
            }
            .metric-item.highlight .m-val {
                color: #00bcd4;
                font-size: 15px;
            }
            .calib-footer {
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-top: 1px solid #28283a;
                padding-top: 16px;
            }
            .calib-btn-primary {
                background: #00bcd4;
                color: #000;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: 700;
                cursor: pointer;
            }
            .calib-btn-primary:hover { background: #26c6da; }
            .calib-btn-secondary {
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.15);
                color: #cbd5e1;
                border-radius: 8px;
                padding: 9px 16px;
                font-size: 12px;
                cursor: pointer;
            }
            .calib-btn-secondary:hover { background: rgba(255,255,255,0.1); color: #fff; }
            .preset-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 10px;
                margin-bottom: 16px;
            }
            .preset-btn {
                background: #181824;
                border: 1px solid #2d2d42;
                border-radius: 8px;
                padding: 12px;
                text-align: left;
                color: #e2e8f0;
                cursor: pointer;
                transition: all 0.15s ease;
            }
            .preset-btn:hover {
                border-color: #00bcd4;
                background: #1e1e2d;
            }
            .preset-btn span {
                font-size: 11px;
                color: #94a3b8;
            }
            .blindspot-stage {
                position: relative;
                height: 120px;
                background: #0a0a10;
                border-radius: 8px;
                border: 1px solid #2d2d42;
                margin-bottom: 14px;
                overflow: hidden;
            }
            .blindspot-cross {
                position: absolute;
                left: 40px;
                top: 50%;
                transform: translateY(-50%);
                color: #ef4444;
                font-size: 28px;
                font-weight: bold;
                line-height: 1;
                user-select: none;
            }
            .blindspot-dot {
                position: absolute;
                left: 100px;
                top: 50%;
                transform: translateY(-50%);
                width: 14px;
                height: 14px;
                background: #f8fafc;
                border-radius: 50%;
            }
            .blindspot-actions {
                display: flex;
                align-items: center;
                gap: 14px;
            }
            .calib-btn-action {
                background: #00bcd4;
                color: #000;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: bold;
                cursor: pointer;
            }
        `;
        document.head.appendChild(style);
    }

    _bindModalEvents() {
        const modal = this.modalElement;
        const slider = modal.querySelector("#sliderCardWidth");
        const distInput = modal.querySelector("#inputViewingDist");

        // Slider listener
        slider.addEventListener("input", (e) => {
            this.currentCardWidthPx = parseFloat(e.target.value);
            this.updateModalUI();
        });

        // Step buttons
        modal.querySelector("#btnCardMinus5").addEventListener("click", () => {
            this.currentCardWidthPx = Math.max(180, this.currentCardWidthPx - 5);
            slider.value = this.currentCardWidthPx;
            this.updateModalUI();
        });
        modal.querySelector("#btnCardMinus1").addEventListener("click", () => {
            this.currentCardWidthPx = Math.max(180, this.currentCardWidthPx - 1);
            slider.value = this.currentCardWidthPx;
            this.updateModalUI();
        });
        modal.querySelector("#btnCardPlus1").addEventListener("click", () => {
            this.currentCardWidthPx = Math.min(600, this.currentCardWidthPx + 1);
            slider.value = this.currentCardWidthPx;
            this.updateModalUI();
        });
        modal.querySelector("#btnCardPlus5").addEventListener("click", () => {
            this.currentCardWidthPx = Math.min(600, this.currentCardWidthPx + 5);
            slider.value = this.currentCardWidthPx;
            this.updateModalUI();
        });

        // Viewing distance input
        distInput.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            if (!isNaN(val) && val >= 15 && val <= 200) {
                this.viewingDistanceCm = val;
                this.updateModalUI();
            }
        });

        // Tabs
        const tabCardBtn = modal.querySelector("#tabCardBtn");
        const tabBlindSpotBtn = modal.querySelector("#tabBlindSpotBtn");
        const tabPresetsBtn = modal.querySelector("#tabPresetsBtn");

        const calibCardSection = modal.querySelector("#calibCardSection");
        const calibBlindSpotSection = modal.querySelector("#calibBlindSpotSection");
        const calibPresetsSection = modal.querySelector("#calibPresetsSection");

        tabCardBtn.addEventListener("click", () => {
            tabCardBtn.classList.add("active");
            tabBlindSpotBtn.classList.remove("active");
            tabPresetsBtn.classList.remove("active");
            calibCardSection.style.display = "block";
            calibBlindSpotSection.style.display = "none";
            calibPresetsSection.style.display = "none";
            if (this.blindSpotActive) this.stopBlindSpotTest();
        });

        tabBlindSpotBtn.addEventListener("click", () => {
            tabBlindSpotBtn.classList.add("active");
            tabCardBtn.classList.remove("active");
            tabPresetsBtn.classList.remove("active");
            calibCardSection.style.display = "none";
            calibBlindSpotSection.style.display = "block";
            calibPresetsSection.style.display = "none";
        });

        tabPresetsBtn.addEventListener("click", () => {
            tabPresetsBtn.classList.add("active");
            tabCardBtn.classList.remove("active");
            tabBlindSpotBtn.classList.remove("active");
            calibCardSection.style.display = "none";
            calibBlindSpotSection.style.display = "none";
            calibPresetsSection.style.display = "block";
            if (this.blindSpotActive) this.stopBlindSpotTest();
        });

        // Blind spot button
        modal.querySelector("#btnStartBlindSpot").addEventListener("click", () => {
            this.startBlindSpotTest();
        });
        modal.querySelector("#btnMarkBlindSpot").addEventListener("click", () => {
            this.recordBlindSpot();
        });

        // Save & Apply
        modal.querySelector("#btnSaveCalib").addEventListener("click", () => {
            this.saveCalibration("credit_card");
            this.closeModal();
            this._showConfirmationBanner();
        });

        // Reset
        modal.querySelector("#btnResetCalib").addEventListener("click", () => {
            localStorage.removeItem("vision_suite_screen_calibration");
            this.loadStoredCalibration();
            slider.value = this.currentCardWidthPx;
            distInput.value = this.viewingDistanceCm;
            this.updateModalUI();
        });

        // Close & Cancel
        modal.querySelector("#btnCalibClose").addEventListener("click", () => this.closeModal());
        modal.querySelector("#btnCancelCalib").addEventListener("click", () => this.closeModal());
    }

    updateModalUI() {
        if (!this.modalElement) return;

        this.recomputeMetrics();

        const cardPreview = this.modalElement.querySelector("#calibCardPreview");
        if (cardPreview) {
            const h = this.currentCardWidthPx / ScreenCalibrator.CARD_ASPECT_RATIO;
            cardPreview.style.width = `${Math.round(this.currentCardWidthPx)}px`;
            cardPreview.style.height = `${Math.round(h)}px`;
        }

        const mDpi = this.modalElement.querySelector("#metricDpi");
        const mPxCm = this.modalElement.querySelector("#metricPxCm");
        const mDist = this.modalElement.querySelector("#metricDist");
        const mPxDeg = this.modalElement.querySelector("#metricPxDeg");

        if (mDpi) mDpi.innerText = `${this.dpi} PPI`;
        if (mPxCm) mPxCm.innerText = `${this.pxPerCm.toFixed(1)} px/cm`;
        if (mDist) mDist.innerText = `${this.viewingDistanceCm.toFixed(1)} cm`;
        if (mPxDeg) mPxDeg.innerHTML = `<strong>${this.pxPerDeg.toFixed(1)} px/&deg;</strong>`;
    }

    applyPreset(presetKey) {
        const preset = ScreenCalibrator.PRESETS[presetKey];
        if (!preset) return;

        this.pxPerMm = preset.ppi / 25.4;
        this.currentCardWidthPx = Math.round(ScreenCalibrator.CARD_WIDTH_MM * this.pxPerMm);
        this.viewingDistanceCm = preset.defaultDistCm;

        const slider = this.modalElement.querySelector("#sliderCardWidth");
        const distInput = this.modalElement.querySelector("#inputViewingDist");
        if (slider) slider.value = this.currentCardWidthPx;
        if (distInput) distInput.value = this.viewingDistanceCm;

        this.updateModalUI();
    }

    /**
     * Virtual Chinrest: Blind Spot Distance Measurement Test
     */
    startBlindSpotTest() {
        const stage = this.modalElement.querySelector("#blindSpotStage");
        const dot = this.modalElement.querySelector("#blindSpotDot");
        const btnStart = this.modalElement.querySelector("#btnStartBlindSpot");
        const btnMark = this.modalElement.querySelector("#btnMarkBlindSpot");
        const resultText = this.modalElement.querySelector("#blindSpotResultText");

        btnStart.style.display = "none";
        btnMark.style.display = "inline-block";
        resultText.innerText = "Dot is sweeping to the right... Press button when it vanishes!";

        this.blindSpotActive = true;
        let currentX = 80;
        const speed = 1.2; // px per frame
        const maxX = stage.clientWidth - 30;

        const animate = () => {
            if (!this.blindSpotActive) return;
            currentX += speed;
            if (currentX > maxX) currentX = 80;
            dot.style.left = `${currentX}px`;
            this.blindSpotAnimId = requestAnimationFrame(animate);
        };

        this.blindSpotAnimId = requestAnimationFrame(animate);
    }

    stopBlindSpotTest() {
        this.blindSpotActive = false;
        if (this.blindSpotAnimId) {
            cancelAnimationFrame(this.blindSpotAnimId);
            this.blindSpotAnimId = null;
        }
        const btnStart = this.modalElement.querySelector("#btnStartBlindSpot");
        const btnMark = this.modalElement.querySelector("#btnMarkBlindSpot");
        if (btnStart) btnStart.style.display = "inline-block";
        if (btnMark) btnMark.style.display = "none";
    }

    recordBlindSpot() {
        if (!this.blindSpotActive) return;
        this.stopBlindSpotTest();

        const stage = this.modalElement.querySelector("#blindSpotStage");
        const dot = this.modalElement.querySelector("#blindSpotDot");
        const cross = this.modalElement.querySelector("#blindSpotCross");
        const resultText = this.modalElement.querySelector("#blindSpotResultText");

        const crossX = cross.offsetLeft;
        const dotX = dot.offsetLeft;
        const deltaPx = Math.abs(dotX - crossX);

        // Convert pixels to mm using the card calibration
        const deltaMm = deltaPx / this.pxPerMm;

        // Viewing distance geometry: d = deltaMm / tan(15.5 deg)
        const radEcc = (ScreenCalibrator.BLIND_SPOT_ECCENTRICITY_DEG * Math.PI) / 180.0;
        const measuredDistCm = (deltaMm / Math.tan(radEcc)) / 10.0;

        if (measuredDistCm >= 25 && measuredDistCm <= 120) {
            this.viewingDistanceCm = Math.round(measuredDistCm * 10) / 10;
            const distInput = this.modalElement.querySelector("#inputViewingDist");
            if (distInput) distInput.value = this.viewingDistanceCm;
            this.updateModalUI();
            resultText.innerText = `✓ Calculated Viewing Distance: ${this.viewingDistanceCm} cm (delta: ${Math.round(deltaMm)} mm)`;
        } else {
            resultText.innerText = `Measurement outside typical range (${measuredDistCm.toFixed(1)} cm). Try again with steady fixation.`;
        }
    }

    _showConfirmationBanner() {
        let banner = document.getElementById("calibConfirmationBanner");
        if (!banner) {
            banner = document.createElement("div");
            banner.id = "calibConfirmationBanner";
            banner.style.cssText = `
                position: fixed;
                bottom: 24px;
                right: 24px;
                background: #00e676;
                color: #000;
                font-weight: 700;
                font-size: 13px;
                padding: 12px 20px;
                border-radius: 8px;
                box-shadow: 0 8px 24px rgba(0, 230, 118, 0.4);
                z-index: 10000;
                display: flex;
                align-items: center;
                gap: 10px;
                animation: fadeInOut 3s ease forwards;
            `;
            document.body.appendChild(banner);
        }
        banner.innerHTML = `✓ Calibration Applied: <strong>${this.dpi} DPI</strong> &bull; <strong>${this.pxPerDeg.toFixed(1)} px/&deg;</strong> at ${this.viewingDistanceCm} cm`;
        banner.style.display = "flex";
        setTimeout(() => { if (banner) banner.style.display = "none"; }, 3500);
    }
}

// Global initialization
if (typeof window !== "undefined") {
    window.ScreenCalibrator = ScreenCalibrator;
    window.screenCalibrator = new ScreenCalibrator();
}

if (typeof module !== "undefined" && module.exports) {
    module.exports = { ScreenCalibrator };
}
