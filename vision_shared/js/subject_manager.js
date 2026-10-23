/**
 * subject_manager.js - Unified Subject & Observer Identifier Manager
 * 
 * Provides:
 * 1. Sensible sequential auto-generation (OBS_01, OBS_02, OBS_03...) without cryptic random hashes.
 * 2. Multi-user tracking & persistent selection of existing subjects across all psychophysics tasks.
 * 3. Bidirectional synchronization via localStorage ('vision_suite_subject_code' & 'vision_suite_known_subjects').
 */

class SubjectManager {
    constructor() {
        this.storageKeyActive = "vision_suite_subject_code";
        this.storageKeyKnown = "vision_suite_known_subjects";
        this.defaultPrefix = "OBS_";
    }

    /**
     * Retrieve list of all known/existing subjects
     * @returns {string[]}
     */
    getKnownSubjects() {
        const subjects = new Set();

        // 1. Read from dedicated known subjects list
        try {
            const raw = localStorage.getItem(this.storageKeyKnown);
            if (raw) {
                const parsed = JSON.parse(raw);
                if (Array.isArray(parsed)) {
                    parsed.forEach(s => {
                        const clean = this._cleanCode(s);
                        if (clean) subjects.add(clean);
                    });
                }
            }
        } catch (e) {}

        // 2. Scan localStorage keys for past session datasets
        try {
            for (let i = 0; i < localStorage.length; i++) {
                const k = localStorage.key(i);
                if (!k) continue;
                if (k.startsWith("vision_suite_csf_") || k.startsWith("vision_suite_dipper_") || k.startsWith("vision_suite_dog_")) {
                    const candidate = k.replace(/^vision_suite_(csf|dipper|dog)_/, "");
                    const clean = this._cleanCode(candidate);
                    if (clean) subjects.add(clean);
                }
            }
        } catch (e) {}

        // 3. Check currently active subject
        try {
            const active = localStorage.getItem(this.storageKeyActive);
            const clean = this._cleanCode(active);
            if (clean) subjects.add(clean);
        } catch (e) {}

        // 4. Always ensure sensible baseline seed subjects exist
        ["OBS_01", "PILOT_01", "BENCHMARK"].forEach(s => subjects.add(s));

        // Return sorted array (OBS_01, OBS_02, etc. first, then others)
        const sorted = Array.from(subjects).sort((a, b) => {
            if (a.startsWith("OBS_") && b.startsWith("OBS_")) {
                const numA = parseInt(a.replace("OBS_", ""), 10) || 0;
                const numB = parseInt(b.replace("OBS_", ""), 10) || 0;
                return numA - numB;
            }
            if (a.startsWith("OBS_")) return -1;
            if (b.startsWith("OBS_")) return 1;
            return a.localeCompare(b);
        });

        try {
            localStorage.setItem(this.storageKeyKnown, JSON.stringify(sorted));
        } catch (e) {}

        return sorted;
    }

    /**
     * Add a subject to known subjects list
     * @param {string} code 
     * @returns {string|null} Cleaned code
     */
    addKnownSubject(code) {
        const clean = this._cleanCode(code);
        if (!clean) return null;

        const known = this.getKnownSubjects();
        if (!known.includes(clean)) {
            known.push(clean);
            try {
                localStorage.setItem(this.storageKeyKnown, JSON.stringify(known));
            } catch (e) {}
        }
        return clean;
    }

    /**
     * Get currently active subject code
     * @returns {string}
     */
    getActiveSubject() {
        // Priority 1: URL Query parameter (?subject=...)
        if (typeof window !== "undefined" && window.location && window.location.search) {
            const urlParams = new URLSearchParams(window.location.search);
            const urlSubj = this._cleanCode(urlParams.get("subject"));
            if (urlSubj) {
                this.setActiveSubject(urlSubj);
                return urlSubj;
            }
        }

        // Priority 2: Stored active code in localStorage
        try {
            const stored = this._cleanCode(localStorage.getItem(this.storageKeyActive));
            if (stored) {
                this.addKnownSubject(stored);
                return stored;
            }
        } catch (e) {}

        // Priority 3: First available known subject, or sensible default OBS_01
        const known = this.getKnownSubjects();
        const fallback = known.length > 0 ? known[0] : "OBS_01";
        this.setActiveSubject(fallback);
        return fallback;
    }

    /**
     * Set active subject code and broadcast change
     * @param {string} code 
     * @returns {string}
     */
    setActiveSubject(code) {
        const clean = this._cleanCode(code) || "OBS_01";
        try {
            localStorage.setItem(this.storageKeyActive, clean);
        } catch (e) {}
        this.addKnownSubject(clean);

        if (typeof window !== "undefined" && typeof window.dispatchEvent === "function") {
            window.dispatchEvent(new CustomEvent("visionSuiteSubjectChanged", {
                detail: { subjectCode: clean }
            }));
        }
        return clean;
    }

    /**
     * Generate a sensible sequential subject code (e.g. OBS_01, OBS_02, OBS_03...)
     * Avoids cryptic random hex hashes unless explicitly asked.
     * @param {string} [prefix="OBS_"]
     * @returns {string}
     */
    generateSensibleSubjectCode(prefix = "OBS_") {
        const known = this.getKnownSubjects();

        // Find existing numbers matching prefix + number
        let maxNum = 0;
        const prefixUpper = prefix.toUpperCase();
        for (const item of known) {
            if (item.startsWith(prefixUpper)) {
                const rest = item.substring(prefixUpper.length);
                const num = parseInt(rest, 10);
                if (!isNaN(num) && num > maxNum) {
                    maxNum = num;
                }
            }
        }

        const nextNum = maxNum + 1;
        const nextCode = `${prefixUpper}${String(nextNum).padStart(2, "0")}`;
        this.setActiveSubject(nextCode);
        return nextCode;
    }

    /**
     * Clean and format subject code
     * @private
     */
    _cleanCode(code) {
        if (!code || typeof code !== "string") return null;
        const trimmed = code.trim().toUpperCase().replace(/[^A-Z0-9_\-]/g, "");
        return (trimmed.length >= 2) ? trimmed : null;
    }

    /**
     * Automatically bind UI controls (Input field, Auto-Gen button, and Select Existing dropdown)
     * @param {Object} options
     * @param {string|HTMLInputElement} options.inputEl
     * @param {string|HTMLSelectElement} [options.selectEl]
     * @param {string|HTMLButtonElement} [options.btnGenEl]
     * @param {Function} [options.onSubjectChanged]
     */
    bindUi(options = {}) {
        const input = typeof options.inputEl === "string" ? document.getElementById(options.inputEl) : options.inputEl;
        const select = typeof options.selectEl === "string" ? document.getElementById(options.selectEl) : options.selectEl;
        const btnGen = typeof options.btnGenEl === "string" ? document.getElementById(options.btnGenEl) : options.btnGenEl;

        const populateSelect = (activeCode) => {
            if (!select) return;
            select.innerHTML = "";

            const defaultOpt = document.createElement("option");
            defaultOpt.value = "";
            defaultOpt.disabled = true;
            defaultOpt.innerText = `👤 Select Existing User (${this.getKnownSubjects().length})`;
            select.appendChild(defaultOpt);

            const known = this.getKnownSubjects();
            let matched = false;
            for (const s of known) {
                const opt = document.createElement("option");
                opt.value = s;
                opt.innerText = (s === activeCode) ? `✓ ${s} (Active)` : s;
                if (s === activeCode) {
                    opt.selected = true;
                    matched = true;
                }
                select.appendChild(opt);
            }
            if (!matched && activeCode) {
                const opt = document.createElement("option");
                opt.value = activeCode;
                opt.innerText = `✓ ${activeCode} (Active)`;
                opt.selected = true;
                select.appendChild(opt);
            }
        };

        const currentActive = this.getActiveSubject();

        if (input) {
            input.value = currentActive;
            input.addEventListener("input", () => {
                const clean = this._cleanCode(input.value);
                if (clean) {
                    this.setActiveSubject(clean);
                    populateSelect(clean);
                    if (options.onSubjectChanged) options.onSubjectChanged(clean);
                }
            });
            input.addEventListener("change", () => {
                const clean = this._cleanCode(input.value);
                if (clean) {
                    input.value = clean;
                    this.setActiveSubject(clean);
                    populateSelect(clean);
                    if (options.onSubjectChanged) options.onSubjectChanged(clean);
                }
            });
        }

        populateSelect(currentActive);

        if (select) {
            select.addEventListener("change", () => {
                const chosen = select.value;
                if (chosen) {
                    this.setActiveSubject(chosen);
                    if (input) input.value = chosen;
                    populateSelect(chosen);
                    if (options.onSubjectChanged) options.onSubjectChanged(chosen);
                }
            });
        }

        if (btnGen) {
            btnGen.addEventListener("click", (e) => {
                if (e && e.preventDefault) e.preventDefault();
                const newCode = this.generateSensibleSubjectCode(this.defaultPrefix);
                if (input) {
                    input.value = newCode;
                    input.classList.add("pulse-highlight");
                    setTimeout(() => input.classList.remove("pulse-highlight"), 600);
                }
                populateSelect(newCode);
                if (options.onSubjectChanged) options.onSubjectChanged(newCode);
            });
        }

        // Listen for cross-module or cross-window subject updates
        if (typeof window !== "undefined") {
            window.addEventListener("visionSuiteSubjectChanged", (e) => {
                const newCode = e.detail?.subjectCode;
                if (newCode) {
                    if (input && input.value !== newCode) input.value = newCode;
                    populateSelect(newCode);
                }
            });
        }
    }
}

// Global Singleton
if (typeof window !== "undefined") {
    window.SubjectManager = SubjectManager;
    window.subjectManager = new SubjectManager();
}

if (typeof module !== "undefined" && module.exports) {
    module.exports = { SubjectManager };
}
