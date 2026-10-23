/**
 * webgl_stimulus_engine.js - High-Precision WebGL2 Hardware Stimulus Engine
 * 
 * Provides:
 * 1. Sub-millisecond VSync-locked frame rendering via requestAnimationFrame (no setTimeout drift).
 * 2. 32-bit floating-point GLSL fragment shaders for Gabors and Difference of Gaussians (DoG).
 * 3. GPU-level Triangular Probability Density Function (TPDF) spatial noise dithering for sub-LSB contrast (< 0.1%).
 * 4. Display gamma linearization (pow(L, 1.0 / gamma)) using empirically calibrated gamma.
 * 5. Hardware-accelerated bounding box clipping and strict non-overlapping outer fixation crosshairs.
 * 6. High-precision timing telemetry: measures exact frame count, jitter (std dev), and dropped frames.
 */

class WebGLStimulusEngine {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.gl = null;
        this.isWebGL2 = false;
        this.gamma = options.gamma || this.getStoredGamma();
        this.enableDithering = options.enableDithering !== undefined ? options.enableDithering : true;
        this.bgLuminance = options.bgLuminance !== undefined ? options.bgLuminance : 128; // 0-255 scale
        this.measuredFps = options.measuredFps || 120.0; // Default Surface Pro 11 refresh rate

        // Shaders and programs
        this.gaborProgram = null;
        this.dogProgram = null;
        this.overlayProgram = null;
        this.quadVBO = null;

        // Presentation runner state
        this.animFrameId = null;
        this.presentationActive = false;

        if (this.canvas && typeof this.canvas.getContext === "function") {
            this._initGL();
        }
    }

    /**
     * Retrieve stored gamma from perceptual contrast calibration or fallback to 2.2
     */
    getStoredGamma() {
        if (typeof localStorage !== "undefined") {
            try {
                const stored = localStorage.getItem("vision_suite_contrast_calibration");
                if (stored) {
                    const parsed = JSON.parse(stored);
                    if (parsed.gamma && parsed.gamma > 0.5 && parsed.gamma < 4.0) {
                        return parsed.gamma;
                    }
                }
            } catch (e) {
                console.warn("Could not load stored gamma calibration:", e);
            }
        }
        return 2.2; // Standard sRGB default
    }

    setGamma(gamma) {
        if (gamma > 0.5 && gamma < 4.0) {
            this.gamma = gamma;
        }
    }

    setMeasuredFps(fps) {
        if (fps >= 30.0 && fps <= 360.0) {
            this.measuredFps = fps;
        }
    }

    /**
     * Initialize WebGL2 context with high-performance parameters
     */
    _initGL() {
        const glAttrs = {
            alpha: false,
            antialias: false, // Invariant pixel geometry
            depth: false,
            stencil: false,
            preserveDrawingBuffer: true,
            powerPreference: "high-performance",
            desynchronized: false
        };

        this.gl = this.canvas.getContext("webgl2", glAttrs);
        if (this.gl) {
            this.isWebGL2 = true;
        } else {
            console.warn("WebGL2 not available, attempting WebGL1 fallback");
            this.gl = this.canvas.getContext("webgl", glAttrs) || this.canvas.getContext("experimental-webgl", glAttrs);
            this.isWebGL2 = false;
        }

        if (!this.gl) {
            console.error("WebGL initialization failed entirely. Will fallback to 2D canvas.");
            return;
        }

        this._initBuffers();
        this._initShaders();
    }

    /**
     * Set up full-screen quad vertex buffer
     */
    _initBuffers() {
        const gl = this.gl;
        // Full screen quad [-1, -1] to [1, 1]
        const vertices = new Float32Array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
            -1.0,  1.0,
             1.0, -1.0,
             1.0,  1.0
        ]);

        this.quadVBO = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
        gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    }

    /**
     * Compile vertex and fragment shaders
     */
    _initShaders() {
        const gl = this.gl;

        // Common Vertex Shader
        const vsSource = `#version 300 es
        in vec2 a_position;
        out vec2 v_texCoord;
        void main() {
            // Map [-1, 1] clip space to [0, 1] texture coordinates
            v_texCoord = (a_position + 1.0) * 0.5;
            gl_Position = vec4(a_position, 0.0, 1.0);
        }`;

        // Fallback WebGL1 Vertex Shader
        const vsSourceGL1 = `
        attribute vec2 a_position;
        varying vec2 v_texCoord;
        void main() {
            v_texCoord = (a_position + 1.0) * 0.5;
            gl_Position = vec4(a_position, 0.0, 1.0);
        }`;

        // 1. GABOR Fragment Shader
        const fsGaborSource = `#version 300 es
        precision highp float;
        out vec4 fragColor;

        uniform vec2 u_resolution;
        uniform vec2 u_center;
        uniform float u_spatial_freq_pixels;
        uniform float u_contrast;
        uniform float u_orientation_rad;
        uniform float u_phase_rad;
        uniform float u_sigma_pixels;
        uniform float u_bg_luminance;
        uniform float u_gamma;
        uniform int u_dither_enabled;
        uniform vec4 u_clip_box; // [minX, minY, maxX, maxY] in window coords

        // High-precision pseudo-random noise for TPDF dithering
        float rand1(vec2 co) {
            return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453123);
        }
        float rand2(vec2 co) {
            return fract(cos(dot(co, vec2(39.346, 11.135))) * 24634.6345678);
        }

        void main() {
            // Fragment coordinate in pixels (account for inverted Y in WebGL)
            vec2 fragCoord = vec2(gl_FragCoord.x, u_resolution.y - gl_FragCoord.y);

            // Background baseline
            float bgNorm = u_bg_luminance / 255.0;
            float linearLum = bgNorm;

            // Check spatial bounding box isolation
            bool insideBox = (fragCoord.x >= u_clip_box.x && fragCoord.x <= u_clip_box.z &&
                              fragCoord.y >= u_clip_box.y && fragCoord.y <= u_clip_box.w);

            if (insideBox && u_contrast > 0.00001) {
                vec2 d = fragCoord - u_center;
                float cosTheta = cos(u_orientation_rad);
                float sinTheta = sin(u_orientation_rad);

                // Rotated coordinates (xPrime aligns with carrier modulation)
                float xPrime = d.x * cosTheta + d.y * sinTheta;
                float yPrime = -d.x * sinTheta + d.y * cosTheta;

                // 2D Gaussian envelope
                float r2 = xPrime * xPrime + yPrime * yPrime;
                float twoSigma2 = 2.0 * u_sigma_pixels * u_sigma_pixels;
                float envelope = exp(-r2 / twoSigma2);

                // Sinusoidal carrier
                float twoPi = 6.28318530718;
                float carrier = cos(twoPi * u_spatial_freq_pixels * xPrime + u_phase_rad);

                // Modulated linear luminance
                linearLum = bgNorm * (1.0 + u_contrast * envelope * carrier);
            }

            // High-Frequency Triangular PDF Spatial Noise Dithering
            // Achieves continuous sub-LSB contrast resolution (< 0.1%) on 8-bit displays
            if (u_dither_enabled == 1) {
                float r1 = rand1(gl_FragCoord.xy);
                float r2 = rand2(gl_FragCoord.xy);
                float tpdf = (r1 + r2 - 1.0) * (0.707 / 255.0);
                linearLum += tpdf;
            }

            linearLum = clamp(linearLum, 0.0, 1.0);

            // Display Gamma Linearization: L_display = pow(L_linear, 1.0 / gamma)
            float displayLum = pow(linearLum, 1.0 / max(0.5, u_gamma));

            fragColor = vec4(displayLum, displayLum, displayLum, 1.0);
        }`;

        // 2. DIFFERENCE OF GAUSSIANS (DoG) Fragment Shader (Dipper & Polarity)
        const fsDogSource = `#version 300 es
        precision highp float;
        out vec4 fragColor;

        uniform vec2 u_resolution;
        uniform vec2 u_center;
        uniform float u_sigma_c;
        uniform float u_sigma_s;
        uniform float u_pedestal_contrast;
        uniform float u_increment_contrast;
        uniform float u_polarity; // +1.0 for ON, -1.0 for OFF
        uniform float u_bg_luminance;
        uniform float u_gamma;
        uniform int u_dither_enabled;
        uniform vec4 u_clip_box;

        float rand1(vec2 co) {
            return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453123);
        }
        float rand2(vec2 co) {
            return fract(cos(dot(co, vec2(39.346, 11.135))) * 24634.6345678);
        }

        void main() {
            vec2 fragCoord = vec2(gl_FragCoord.x, u_resolution.y - gl_FragCoord.y);
            float bgNorm = u_bg_luminance / 255.0;
            float linearLum = bgNorm;

            bool insideBox = (fragCoord.x >= u_clip_box.x && fragCoord.x <= u_clip_box.z &&
                              fragCoord.y >= u_clip_box.y && fragCoord.y <= u_clip_box.w);

            float totalContrast = u_pedestal_contrast + u_increment_contrast;

            if (insideBox && totalContrast > 0.00001) {
                vec2 d = fragCoord - u_center;
                float r2 = d.x * d.x + d.y * d.y;

                float twoSigmaC2 = 2.0 * u_sigma_c * u_sigma_c;
                float twoSigmaS2 = 2.0 * u_sigma_s * u_sigma_s;

                // Zero-DC balanced surround ratio
                float balanceRatio = (u_sigma_c * u_sigma_c) / (u_sigma_s * u_sigma_s);
                float peakNorm = 1.0 / (1.0 - balanceRatio);

                float gCenter = exp(-r2 / twoSigmaC2);
                float gSurround = exp(-r2 / twoSigmaS2);

                float dogWave = (gCenter - balanceRatio * gSurround) * peakNorm;
                float deltaC = u_polarity * totalContrast * dogWave;

                linearLum = bgNorm * (1.0 + deltaC);
            }

            if (u_dither_enabled == 1) {
                float r1 = rand1(gl_FragCoord.xy);
                float r2 = rand2(gl_FragCoord.xy);
                float tpdf = (r1 + r2 - 1.0) * (0.707 / 255.0);
                linearLum += tpdf;
            }

            linearLum = clamp(linearLum, 0.0, 1.0);
            float displayLum = pow(linearLum, 1.0 / max(0.5, u_gamma));

            fragColor = vec4(displayLum, displayLum, displayLum, 1.0);
        }`;

        // 3. OVERLAY SHADER for Bounding Box Border and Non-Overlapping Fixation Crosshairs
        const fsOverlaySource = `#version 300 es
        precision highp float;
        out vec4 fragColor;

        uniform vec2 u_resolution;
        uniform vec2 u_center;
        uniform vec4 u_clip_box;
        uniform vec4 u_box_color;
        uniform vec4 u_arm_color;
        uniform float u_border_width;
        uniform float u_arm_len;
        uniform float u_gap;
        uniform vec2 u_fixation_center;
        uniform int u_draw_central_cross;
        uniform int u_show_bounding_box;

        void main() {
            vec2 fragCoord = vec2(gl_FragCoord.x, u_resolution.y - gl_FragCoord.y);

            vec4 color = vec4(0.0, 0.0, 0.0, 0.0); // Transparent by default
            float bw = u_border_width;
            float armThickness = 2.5;
            float halfArm = armThickness * 0.5;

            if (u_show_bounding_box == 1) {
                // Bounding box border
                bool onLeftEdge = (abs(fragCoord.x - u_clip_box.x) <= bw * 0.5 && fragCoord.y >= u_clip_box.y - bw && fragCoord.y <= u_clip_box.w + bw);
                bool onRightEdge = (abs(fragCoord.x - u_clip_box.z) <= bw * 0.5 && fragCoord.y >= u_clip_box.y - bw && fragCoord.y <= u_clip_box.w + bw);
                bool onTopEdge = (abs(fragCoord.y - u_clip_box.y) <= bw * 0.5 && fragCoord.x >= u_clip_box.x - bw && fragCoord.x <= u_clip_box.z + bw);
                bool onBottomEdge = (abs(fragCoord.y - u_clip_box.w) <= bw * 0.5 && fragCoord.x >= u_clip_box.x - bw && fragCoord.x <= u_clip_box.z + bw);

                if (onLeftEdge || onRightEdge || onTopEdge || onBottomEdge) {
                    color = u_box_color;
                }

                // 4 Outer Non-Overlapping Fixation Crosshair Arms (strictly OUTSIDE bounding box)
                // Top Arm
                bool onTopArm = (abs(fragCoord.x - u_center.x) <= halfArm &&
                                 fragCoord.y >= (u_clip_box.y - u_gap - u_arm_len) &&
                                 fragCoord.y <= (u_clip_box.y - u_gap));

                // Bottom Arm
                bool onBottomArm = (abs(fragCoord.x - u_center.x) <= halfArm &&
                                    fragCoord.y >= (u_clip_box.w + u_gap) &&
                                    fragCoord.y <= (u_clip_box.w + u_gap + u_arm_len));

                // Left Arm
                bool onLeftArm = (abs(fragCoord.y - u_center.y) <= halfArm &&
                                  fragCoord.x >= (u_clip_box.x - u_gap - u_arm_len) &&
                                  fragCoord.x <= (u_clip_box.x - u_gap));

                // Right Arm
                bool onRightArm = (abs(fragCoord.y - u_center.y) <= halfArm &&
                                   fragCoord.x >= (u_clip_box.z + u_gap) &&
                                   fragCoord.x <= (u_clip_box.z + u_gap + u_arm_len));

                if (onTopArm || onBottomArm || onLeftArm || onRightArm) {
                    color = u_arm_color;
                }
            }

            // Central Fixation Cross (for peripheral presentations or when bounding box is disabled)
            if (u_draw_central_cross == 1) {
                float fixArmLen = 10.0;
                bool onCentH = (abs(fragCoord.y - u_fixation_center.y) <= halfArm && abs(fragCoord.x - u_fixation_center.x) <= fixArmLen);
                bool onCentV = (abs(fragCoord.x - u_fixation_center.x) <= halfArm && abs(fragCoord.y - u_fixation_center.y) <= fixArmLen);
                if (onCentH || onCentV) {
                    color = u_arm_color;
                }
            }

            if (color.a < 0.01) {
                discard;
            }
            fragColor = color;
        }`;

        // Compile Programs
        if (this.isWebGL2) {
            this.gaborProgram = this._buildProgram(vsSource, fsGaborSource);
            this.dogProgram = this._buildProgram(vsSource, fsDogSource);
            this.overlayProgram = this._buildProgram(vsSource, fsOverlaySource);
        }
    }

    _compileShader(source, type) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error("Shader compilation error:", gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    _buildProgram(vsSource, fsSource) {
        const gl = this.gl;
        const vs = this._compileShader(vsSource, gl.VERTEX_SHADER);
        const fs = this._compileShader(fsSource, gl.FRAGMENT_SHADER);
        if (!vs || !fs) return null;

        const program = gl.createProgram();
        gl.attachShader(program, vs);
        gl.attachShader(program, fs);
        gl.linkProgram(program);

        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error("Program link error:", gl.getProgramInfoLog(program));
            gl.deleteProgram(program);
            return null;
        }
        return program;
    }

    /**
     * Clear canvas with mid-gray background
     */
    clearBackground(customLum = null) {
        const gl = this.gl;
        if (!gl) return;
        const lum = customLum !== null ? customLum : this.bgLuminance;
        const norm = Math.pow(lum / 255.0, 1.0 / max(0.5, this.gamma));
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.clearColor(norm, norm, norm, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);
    }

    /**
     * Render Gabor patch using WebGL2
     */
    renderGabor(params) {
        const gl = this.gl;
        if (!gl || !this.gaborProgram) return;

        const {
            centerX,
            centerY,
            spatialFreqPixels,
            contrast,
            orientationRad = 0,
            phaseRad = 0,
            sigmaPixels,
            boxHalfSizePx = 60,
            feedback = null,
            showBoundingBox = true,
            fixationCenter = null
        } = params;

        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.useProgram(this.gaborProgram);

        // Bind Quad
        const posLoc = gl.getAttribLocation(this.gaborProgram, "a_position");
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

        // Set Uniforms
        gl.uniform2f(gl.getUniformLocation(this.gaborProgram, "u_resolution"), this.canvas.width, this.canvas.height);
        gl.uniform2f(gl.getUniformLocation(this.gaborProgram, "u_center"), centerX, centerY);
        gl.uniform1f(gl.getUniformLocation(this.gaborProgram, "u_spatial_freq_pixels"), spatialFreqPixels);
        gl.uniform1f(gl.getUniformLocation(this.gaborProgram, "u_contrast"), contrast);
        gl.uniform1f(gl.getUniformLocation(this.gaborProgram, "u_orientation_rad"), orientationRad);
        gl.uniform1f(gl.getUniformLocation(this.gaborProgram, "u_phase_rad"), phaseRad);
        gl.uniform1f(gl.getUniformLocation(this.gaborProgram, "u_sigma_pixels"), sigmaPixels);
        gl.uniform1f(gl.getUniformLocation(this.gaborProgram, "u_bg_luminance"), this.bgLuminance);
        gl.uniform1f(gl.getUniformLocation(this.gaborProgram, "u_gamma"), this.gamma);
        gl.uniform1i(gl.getUniformLocation(this.gaborProgram, "u_dither_enabled"), this.enableDithering ? 1 : 0);

        let minX = 0, minY = 0, maxX = 0, maxY = 0;
        if (showBoundingBox && boxHalfSizePx) {
            minX = centerX - boxHalfSizePx;
            minY = centerY - boxHalfSizePx;
            maxX = centerX + boxHalfSizePx;
            maxY = centerY + boxHalfSizePx;
            gl.uniform4f(gl.getUniformLocation(this.gaborProgram, "u_clip_box"), minX, minY, maxX, maxY);
        } else {
            gl.uniform4f(gl.getUniformLocation(this.gaborProgram, "u_clip_box"), -10000.0, -10000.0, 100000.0, 100000.0);
        }

        gl.drawArrays(gl.TRIANGLES, 0, 6);

        // Render Bounding Box and Fixation Overlay
        this.renderOverlay({
            centerX,
            centerY,
            minX,
            minY,
            maxX,
            maxY,
            feedback,
            fixationCenter,
            showBoundingBox
        });
    }

    /**
     * Render Difference of Gaussians (DoG) patch (Dipper & Polarity)
     */
    renderDog(params) {
        const gl = this.gl;
        if (!gl || !this.dogProgram) return;

        const {
            centerX,
            centerY,
            sigmaCPixels,
            sigmaSPixels,
            pedestalContrast = 0.0,
            incrementContrast = 0.0,
            polarity = "ON", // "ON" or "OFF"
            boxHalfSizePx = 60,
            feedback = null,
            showBoundingBox = true,
            fixationCenter = null
        } = params;

        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.useProgram(this.dogProgram);

        const posLoc = gl.getAttribLocation(this.dogProgram, "a_position");
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

        gl.uniform2f(gl.getUniformLocation(this.dogProgram, "u_resolution"), this.canvas.width, this.canvas.height);
        gl.uniform2f(gl.getUniformLocation(this.dogProgram, "u_center"), centerX, centerY);
        gl.uniform1f(gl.getUniformLocation(this.dogProgram, "u_sigma_c"), sigmaCPixels);
        gl.uniform1f(gl.getUniformLocation(this.dogProgram, "u_sigma_s"), sigmaSPixels || (sigmaCPixels * 2.0));
        gl.uniform1f(gl.getUniformLocation(this.dogProgram, "u_pedestal_contrast"), pedestalContrast);
        gl.uniform1f(gl.getUniformLocation(this.dogProgram, "u_increment_contrast"), incrementContrast);
        gl.uniform1f(gl.getUniformLocation(this.dogProgram, "u_polarity"), polarity === "ON" ? 1.0 : -1.0);
        gl.uniform1f(gl.getUniformLocation(this.dogProgram, "u_bg_luminance"), this.bgLuminance);
        gl.uniform1f(gl.getUniformLocation(this.dogProgram, "u_gamma"), this.gamma);
        gl.uniform1i(gl.getUniformLocation(this.dogProgram, "u_dither_enabled"), this.enableDithering ? 1 : 0);

        let minX = 0, minY = 0, maxX = 0, maxY = 0;
        if (showBoundingBox && boxHalfSizePx) {
            minX = centerX - boxHalfSizePx;
            minY = centerY - boxHalfSizePx;
            maxX = centerX + boxHalfSizePx;
            maxY = centerY + boxHalfSizePx;
            gl.uniform4f(gl.getUniformLocation(this.dogProgram, "u_clip_box"), minX, minY, maxX, maxY);
        } else {
            // Full canvas bounds, natural Gaussian falloff with no clipping box
            gl.uniform4f(gl.getUniformLocation(this.dogProgram, "u_clip_box"), -10000.0, -10000.0, 100000.0, 100000.0);
        }

        gl.drawArrays(gl.TRIANGLES, 0, 6);

        this.renderOverlay({
            centerX,
            centerY,
            minX,
            minY,
            maxX,
            maxY,
            feedback,
            fixationCenter,
            showBoundingBox
        });
    }

    /**
     * Render Bounding Box and Fixation Guides (with feedback coloring)
     */
    renderOverlay(params) {
        const gl = this.gl;
        if (!gl || !this.overlayProgram) return;

        const {
            centerX,
            centerY,
            minX = 0,
            minY = 0,
            maxX = 0,
            maxY = 0,
            feedback = null,
            fixationCenter = null,
            showBoundingBox = true
        } = params;

        // Feedback colors: green = correct, red = incorrect, dark-gray = neutral
        let boxColor = [0.15, 0.15, 0.15, 0.90];
        let armColor = [0.10, 0.10, 0.10, 0.95];

        if (feedback === "CORRECT") {
            boxColor = [0.18, 0.49, 0.20, 0.95]; // Green #2e7d32
            armColor = [0.18, 0.49, 0.20, 1.00];
        } else if (feedback === "INCORRECT") {
            boxColor = [0.78, 0.16, 0.16, 0.95]; // Red #c62828
            armColor = [0.78, 0.16, 0.16, 1.00];
        }

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        gl.useProgram(this.overlayProgram);
        const posLoc = gl.getAttribLocation(this.overlayProgram, "a_position");
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

        gl.uniform2f(gl.getUniformLocation(this.overlayProgram, "u_resolution"), this.canvas.width, this.canvas.height);
        gl.uniform2f(gl.getUniformLocation(this.overlayProgram, "u_center"), centerX, centerY);
        gl.uniform4f(gl.getUniformLocation(this.overlayProgram, "u_clip_box"), minX, minY, maxX, maxY);
        gl.uniform4fv(gl.getUniformLocation(this.overlayProgram, "u_box_color"), new Float32Array(boxColor));
        gl.uniform4fv(gl.getUniformLocation(this.overlayProgram, "u_arm_color"), new Float32Array(armColor));
        gl.uniform1f(gl.getUniformLocation(this.overlayProgram, "u_border_width"), feedback ? 2.5 : 2.0);
        gl.uniform1f(gl.getUniformLocation(this.overlayProgram, "u_arm_len"), 18.0);
        gl.uniform1f(gl.getUniformLocation(this.overlayProgram, "u_gap"), 6.0);
        gl.uniform1i(gl.getUniformLocation(this.overlayProgram, "u_show_bounding_box"), showBoundingBox ? 1 : 0);

        const hasFixCenter = (fixationCenter && Array.isArray(fixationCenter)) || !showBoundingBox;
        gl.uniform1i(gl.getUniformLocation(this.overlayProgram, "u_draw_central_cross"), hasFixCenter ? 1 : 0);
        if (fixationCenter && Array.isArray(fixationCenter)) {
            gl.uniform2f(gl.getUniformLocation(this.overlayProgram, "u_fixation_center"), fixationCenter[0], fixationCenter[1]);
        } else {
            gl.uniform2f(gl.getUniformLocation(this.overlayProgram, "u_fixation_center"), centerX, centerY);
        }

        gl.drawArrays(gl.TRIANGLES, 0, 6);
        gl.disable(gl.BLEND);
    }

    /**
     * Render neutral fixation display (blank stimulus area inside bounding box, with outer crosshairs, or single central fixation)
     */
    renderFixationOnly(params) {
        const {
            centerX,
            centerY,
            boxHalfSizePx = 60,
            feedback = null,
            fixationCenter = null,
            showBoundingBox = true
        } = params;
        this.clearBackground();
        const minX = centerX - boxHalfSizePx;
        const minY = centerY - boxHalfSizePx;
        const maxX = centerX + boxHalfSizePx;
        const maxY = centerY + boxHalfSizePx;
        this.renderOverlay({
            centerX,
            centerY,
            minX,
            minY,
            maxX,
            maxY,
            feedback,
            fixationCenter: fixationCenter || (showBoundingBox ? null : [centerX, centerY]),
            showBoundingBox
        });
    }

    /**
     * VSync-Locked Stimulus Interval Presentation Engine
     * 
     * Runs presentation for exact target duration using requestAnimationFrame.
     * Computes target frames: Math.round(durationMs / (1000 / measuredFps))
     * E.g. at 120 Hz, 120 ms = exactly 14 frames!
     * Tracks frame timestamps, jitter, and dropped frames.
     * 
     * @param {Object} config
     * @param {string} config.type - "GABOR", "DOG", or "BLANK"
     * @param {Object} config.params - Stimulus parameters
     * @param {number} config.durationMs - Target duration (e.g. 120 ms)
     * @param {Function} config.onComplete - Called when presentation concludes with telemetry stats
     */
    presentInterval(config) {
        const { type, params, durationMs = 120, onComplete = null } = config;

        if (this.presentationActive) {
            cancelAnimationFrame(this.animFrameId);
            this.presentationActive = false;
        }

        const nominalFrameDuration = 1000.0 / this.measuredFps;
        const targetFrames = Math.max(1, Math.round(durationMs / nominalFrameDuration));

        let frameCount = 0;
        const timestamps = [];
        const startTime = performance.now();
        this.presentationActive = true;

        const frameStep = (now) => {
            if (!this.presentationActive) return;

            timestamps.push(now);
            frameCount++;

            // Render Stimulus Frame
            if (type === "GABOR") {
                this.renderGabor(params);
            } else if (type === "DOG") {
                this.renderDog(params);
            } else {
                this.renderFixationOnly(params);
            }

            // Check if duration satisfied
            if (frameCount >= targetFrames) {
                this.presentationActive = false;
                const endTime = performance.now();
                const actualDurationMs = endTime - startTime;

                // Frame-to-frame delta jitter analysis
                const deltas = [];
                let droppedFrames = 0;
                for (let i = 1; i < timestamps.length; i++) {
                    const dt = timestamps[i] - timestamps[i - 1];
                    deltas.push(dt);
                    if (dt > 1.5 * nominalFrameDuration) {
                        droppedFrames++;
                    }
                }

                // Standard deviation of inter-frame intervals
                let jitterStdDev = 0;
                if (deltas.length > 1) {
                    const meanDt = deltas.reduce((a, b) => a + b, 0) / deltas.length;
                    const variance = deltas.reduce((a, b) => a + Math.pow(b - meanDt, 2), 0) / deltas.length;
                    jitterStdDev = Math.sqrt(variance);
                }

                const telemetry = {
                    targetFrames,
                    actualFrames: frameCount,
                    targetDurationMs: durationMs,
                    actualDurationMs: Number(actualDurationMs.toFixed(2)),
                    jitterMs: Number(jitterStdDev.toFixed(3)),
                    droppedFrames,
                    nominalFps: this.measuredFps,
                    timestampFirst: timestamps[0],
                    timestampLast: timestamps[timestamps.length - 1]
                };

                if (onComplete) {
                    onComplete(telemetry);
                }
            } else {
                this.animFrameId = requestAnimationFrame(frameStep);
            }
        };

        this.animFrameId = requestAnimationFrame(frameStep);
    }

    cancelPresentation() {
        if (this.presentationActive) {
            cancelAnimationFrame(this.animFrameId);
            this.presentationActive = false;
        }
    }

    // =========================================================================
    // MATHEMATICAL GROUND TRUTH UTILITIES FOR OFFLINE / NODE.JS VERIFICATION
    // =========================================================================

    static computeGaborLuminance(x, y, params) {
        const {
            centerX = 0,
            centerY = 0,
            spatialFreqPixels,
            contrast,
            orientationRad = 0,
            phaseRad = 0,
            sigmaPixels,
            bgLuminance = 128
        } = params;

        const dx = x - centerX;
        const dy = y - centerY;
        const cosTheta = Math.cos(orientationRad);
        const sinTheta = Math.sin(orientationRad);

        const xPrime = dx * cosTheta + dy * sinTheta;
        const yPrime = -dx * sinTheta + dy * cosTheta;

        const r2 = xPrime * xPrime + yPrime * yPrime;
        const envelope = Math.exp(-r2 / (2.0 * sigmaPixels * sigmaPixels));
        const carrier = Math.cos(2.0 * Math.PI * spatialFreqPixels * xPrime + phaseRad);

        return bgLuminance * (1.0 + contrast * envelope * carrier);
    }

    static computeDogLuminance(x, y, params) {
        const {
            centerX = 0,
            centerY = 0,
            sigmaCPixels,
            sigmaSPixels = sigmaCPixels * 2.0,
            totalContrast,
            polarity = "ON",
            bgLuminance = 128
        } = params;

        const dx = x - centerX;
        const dy = y - centerY;
        const r2 = dx * dx + dy * dy;

        const balanceRatio = Math.pow(sigmaCPixels / sigmaSPixels, 2.0);
        const peakNorm = 1.0 / (1.0 - balanceRatio);

        const gCenter = Math.exp(-r2 / (2.0 * sigmaCPixels * sigmaCPixels));
        const gSurround = Math.exp(-r2 / (2.0 * sigmaSPixels * sigmaSPixels));

        const dogWave = (gCenter - balanceRatio * gSurround) * peakNorm;
        const sign = polarity === "ON" ? 1.0 : -1.0;

        return bgLuminance * (1.0 + sign * totalContrast * dogWave);
    }

    static applyGamma(luminance, gamma) {
        const norm = Math.max(0, Math.min(1.0, luminance / 255.0));
        return Math.pow(norm, 1.0 / Math.max(0.5, gamma)) * 255.0;
    }

    static calculateTargetFrames(durationMs, fps) {
        const frameTime = 1000.0 / fps;
        return Math.max(1, Math.round(durationMs / frameTime));
    }
}

function max(a, b) {
    return Math.max(a, b);
}

if (typeof module !== "undefined" && module.exports) {
    module.exports = { WebGLStimulusEngine };
}
