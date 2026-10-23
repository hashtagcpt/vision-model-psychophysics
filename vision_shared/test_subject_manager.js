const assert = require("assert");

// Mock localStorage for Node.js environment
class LocalStorageMock {
    constructor() {
        this.store = {};
    }
    getItem(key) {
        return this.store[key] || null;
    }
    setItem(key, value) {
        this.store[key] = String(value);
    }
    removeItem(key) {
        delete this.store[key];
    }
    clear() {
        this.store = {};
    }
    key(i) {
        return Object.keys(this.store)[i] || null;
    }
    get length() {
        return Object.keys(this.store).length;
    }
}

global.localStorage = new LocalStorageMock();
global.window = {
    location: { search: "" },
    dispatchEvent: () => {}
};

const { SubjectManager } = require("./js/subject_manager.js");

console.log("==========================================================================");
console.log(" RUNNING AUTOMATED UNIT TESTS: SUBJECT MANAGER (AUTO-GEN & EXISTING USER)");
console.log("==========================================================================");

// Test 1: Sensible Default Generation
console.log("[Test 1] Testing sensible default subject generation on empty state...");
const sm1 = new SubjectManager();
localStorage.clear();
const initialSubj = sm1.getActiveSubject();
console.log(`  -> Initial active subject: ${initialSubj}`);
assert.strictEqual(initialSubj, "OBS_01", "Expected initial subject to default sensibly to OBS_01");
console.log("  [PASS] Initial sensible default verified!");

// Test 2: Sequential Auto-Generation
console.log("\n[Test 2] Testing sequential auto-generation (OBS_01 -> OBS_02 -> OBS_03)...");
const code1 = sm1.generateSensibleSubjectCode("OBS_");
console.log(`  -> Generated 1: ${code1}`);
assert.strictEqual(code1, "OBS_02", "Expected next code to be OBS_02");

const code2 = sm1.generateSensibleSubjectCode("OBS_");
console.log(`  -> Generated 2: ${code2}`);
assert.strictEqual(code2, "OBS_03", "Expected next code to be OBS_03");

const code3 = sm1.generateSensibleSubjectCode("OBS_");
console.log(`  -> Generated 3: ${code3}`);
assert.strictEqual(code3, "OBS_04", "Expected next code to be OBS_04");
console.log("  [PASS] Sequential auto-generation verified!");

// Test 3: Existing User Tracking & Retrieval
console.log("\n[Test 3] Testing tracking of existing users and custom additions...");
sm1.addKnownSubject("PILOT_ALPHA");
sm1.addKnownSubject("SUBJECT_42");
const knownList = sm1.getKnownSubjects();
console.log(`  -> Known subjects: ${JSON.stringify(knownList)}`);
assert(knownList.includes("OBS_01"), "Should include OBS_01");
assert(knownList.includes("OBS_02"), "Should include OBS_02");
assert(knownList.includes("OBS_03"), "Should include OBS_03");
assert(knownList.includes("PILOT_ALPHA"), "Should include PILOT_ALPHA");
assert(knownList.includes("SUBJECT_42"), "Should include SUBJECT_42");
console.log("  [PASS] Known subject tracking verified!");

// Test 4: Selecting Existing User
console.log("\n[Test 4] Testing selection of an existing user...");
sm1.setActiveSubject("OBS_02");
assert.strictEqual(sm1.getActiveSubject(), "OBS_02", "Active subject should be updated to OBS_02");
assert.strictEqual(localStorage.getItem("vision_suite_subject_code"), "OBS_02", "localStorage should match active subject");
console.log("  [PASS] Existing user selection verified!");

// Test 5: Next generation preserves sequence even if arbitrary subjects exist
console.log("\n[Test 5] Testing that next auto-generation detects highest OBS_ number...");
const codeNext = sm1.generateSensibleSubjectCode("OBS_");
console.log(`  -> Next generated after OBS_04 was recorded: ${codeNext}`);
assert.strictEqual(codeNext, "OBS_05", "Expected next to be OBS_05");
console.log("  [PASS] Non-conflicting sequence verified!");

console.log("\n==========================================================================");
console.log(" ALL SUBJECT MANAGER UNIT TESTS PASSED (5/5)");
console.log("==========================================================================");
