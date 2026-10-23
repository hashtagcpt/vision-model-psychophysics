/**
 * data_exporter.js - Vision Science Psychophysics Data Logger & Exporter
 * Formats, summarizes, and exports trial-by-trial logs and QUEST+ estimated thresholds.
 */

class DataExporter {
    constructor() {
        this.trials = [];
        let initialSubj = "OBS_01";
        try {
            if (typeof window !== "undefined" && window.subjectManager) {
                initialSubj = window.subjectManager.getActiveSubject();
            } else if (typeof localStorage !== "undefined") {
                initialSubj = localStorage.getItem("vision_suite_subject_code") || "OBS_01";
            }
        } catch (e) {}
        this.participantId = initialSubj;
        this.startTime = new Date().toISOString();
    }

    setParticipantId(id) {
        if (id && typeof id === "string" && id.trim().length >= 2) {
            this.participantId = id.trim().toUpperCase();
        }
    }

    logTrial(record) {
        this.trials.push({
            participantId: this.participantId,
            timestamp: new Date().toISOString(),
            ...record
        });
    }

    /**
     * Compute final summary table across all conditions
     * @param {Array<Object>} conditionList - Array of condition objects with their quest instances
     * @returns {Array<Object>}
     */
    generateConditionSummary(conditionList) {
        return conditionList.map(cond => {
            const quest = cond.quest;
            const logThreshold = quest.getThresholdMean();
            const linThreshold = quest.getLinearContrastThreshold();
            const thresholdStd = quest.getThresholdStd();
            const finalEntropy = quest.getCurrentEntropy();

            return {
                participantId: this.participantId,
                conditionId: cond.id,
                sizeCenterDeg: cond.sizeDeg,
                eccentricityDeg: cond.eccDeg,
                trialsCompleted: quest.history.length,
                log10Threshold: Number(logThreshold.toFixed(4)),
                linearContrastPercent: Number((linThreshold * 100.0).toFixed(2)),
                contrastSensitivity: Number((1.0 / Math.max(1e-4, linThreshold)).toFixed(1)),
                thresholdStdDev: Number(thresholdStd.toFixed(4)),
                finalEntropyBits: Number(finalEntropy.toFixed(3))
            };
        });
    }

    /**
     * Converts an array of objects to CSV string
     * @param {Array<Object>} rows 
     * @returns {string}
     */
    _toCsv(rows) {
        if (!rows || rows.length === 0) return "";
        const headers = Object.keys(rows[0]);
        const lines = [headers.join(",")];

        for (const row of rows) {
            const values = headers.map(header => {
                const val = row[header];
                if (typeof val === "string" && val.includes(",")) {
                    return `"${val}"`;
                }
                return val !== undefined ? val : "";
            });
            lines.push(values.join(","));
        }
        return lines.join("\n");
    }

    /**
     * Trigger browser file download
     * @param {string} content 
     * @param {string} filename 
     * @param {string} mimeType 
     */
    _downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    downloadTrialDataCsv() {
        const csv = this._toCsv(this.trials);
        const filename = `${this.participantId}_dog_quest_trials.csv`;
        this._downloadFile(csv, filename, "text/csv;charset=utf-8;");
    }

    downloadSummaryCsv(conditionList) {
        const summary = this.generateConditionSummary(conditionList);
        const csv = this._toCsv(summary);
        const filename = `${this.participantId}_dog_quest_thresholds_summary.csv`;
        this._downloadFile(csv, filename, "text/csv;charset=utf-8;");
    }

    downloadFullSessionJson(conditionList) {
        const payload = {
            participantId: this.participantId,
            sessionStart: this.startTime,
            sessionEnd: new Date().toISOString(),
            conditionSummary: this.generateConditionSummary(conditionList),
            trialData: this.trials
        };
        const json = JSON.stringify(payload, null, 2);
        const filename = `${this.participantId}_dog_quest_session.json`;
        this._downloadFile(json, filename, "application/json");
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DataExporter };
}
