const { execSync } = require("child_process");

const edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";

const urls = [
    "http://localhost:8091/index.html",
    "http://localhost:8091/vision_experiment_csf/index.html",
    "http://localhost:8091/vision_experiment_dipper/index.html",
    "http://localhost:8091/vision_experiment_dog_quest/index.html",
    "http://localhost:8091/results_viewer.html"
];

for (const u of urls) {
    try {
        const cmd = `"${edgePath}" --headless=new --dump-dom "${u}"`;
        const res = execSync(cmd, { timeout: 10000, encoding: "utf8" });
        console.log(`URL: ${u}`);
        console.log(`  Rendered length: ${res.length} bytes`);
        console.log(`  Has <title>: ${res.includes("<title>")}`);
    } catch (e) {
        console.error(`Error loading ${u}:`, e.message);
    }
}
