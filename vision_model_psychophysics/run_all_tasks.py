#!/usr/bin/env python3
"""
run_all_tasks.py - Master Evaluator for Expanded Psychophysical Tasks & Predictions
Runs all 8 new psychophysical tasks + 4 novel predictions across all observers:
1. ModelFest 43-Stimulus Suite in Noise (Watson 2000)
2. Letter & Word Inefficiency (Pelli et al. 2003)
3. Face & Letter Bandpass Masking (Gold et al. 1999; Majaj et al. 2002)
4. Compound Harmonic Phase Reversal (Burr 1980; Morrone & Burr 1988)
5. Band-Limited Noise Masking (Henning 1975; Solomon & Pelli 1994)
6. Random-Dot Stereograms & Cyclopean Disparity (Julesz 1971; Cormack 1991)
7. Visual Attention & Search Inefficiency (Palmer 1994; Eckstein 1998)
8. Visual Working Memory Precision (Bays & Husain 2008; Luck & Vogel 1997)
9. Novel Predictions & Model Disagreement Suite (predictions/predict_disagreements.py)

Exports:
  - vision_model_psychophysics/data/comprehensive_task_benchmark.json
  - vision_model_psychophysics/data/comprehensive_task_summary.csv
  - vision_model_psychophysics/js/comprehensive_task_data.js
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

# Add project root and tasks to path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR / "tasks") not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR / "tasks"))
if str(SCRIPT_DIR / "predictions") not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR / "predictions"))

from observers import build_observer_zoo
from stimuli import StimulusConfig
from tasks.modelfest.modelfest_task import evaluate_modelfest_suite
from tasks.letter_word_inefficiency.letter_word_task import evaluate_letter_word_inefficiency_suite
from tasks.face_letter_bandpass.face_letter_task import evaluate_face_letter_bandpass_suite
from tasks.phase_reversal.phase_reversal_task import evaluate_phase_reversal_suite
from tasks.band_limited_noise.band_limited_task import evaluate_band_limited_suite
from tasks.random_dot_stereograms.stereogram_task import evaluate_stereogram_suite
from tasks.visual_attention_search.attention_search_task import evaluate_attention_search_suite
from tasks.working_memory.working_memory_task import evaluate_working_memory_suite
from predictions.predict_disagreements import execute_all_predictions


def execute_comprehensive_suite() -> Dict[str, Any]:
    t0 = time.perf_counter()
    cfg = StimulusConfig()
    zoo = build_observer_zoo(cfg)

    print("=" * 80)
    print(" COMPREHENSIVE VISION PSYCHOPHYSICS & NOVEL PREDICTIONS SUITE")
    print("=" * 80)

    print(" [1/9] Evaluating ModelFest 43-Stimulus Suite in Noise (Watson 2000)...")
    res_modelfest = evaluate_modelfest_suite(zoo)

    print(" [2/9] Evaluating Letter & Word Identification Inefficiency (Pelli et al. 2003)...")
    res_words = evaluate_letter_word_inefficiency_suite(zoo)

    print(" [3/9] Evaluating Face & Letter Bandpass Noise Masking (Gold et al. 1999)...")
    res_bandpass = evaluate_face_letter_bandpass_suite(zoo)

    print(" [4/9] Evaluating Compound Harmonic Phase Reversal (Burr 1980)...")
    res_phase = evaluate_phase_reversal_suite(zoo)

    print(" [5/9] Evaluating Band-Limited Noise Masking (Henning 1975)...")
    res_band_limited = evaluate_band_limited_suite(zoo)

    print(" [6/9] Evaluating Random-Dot Stereograms & Cyclopean Disparity (Julesz 1971)...")
    res_rds = evaluate_stereogram_suite(zoo)

    print(" [7/9] Evaluating Visual Attention & Visual Search Inefficiency (Palmer 1994)...")
    res_attention = evaluate_attention_search_suite(zoo)

    print(" [8/9] Evaluating Visual Working Memory Precision (Bays & Husain 2008)...")
    res_wm = evaluate_working_memory_suite(zoo)

    print(" [9/9] Generating Novel Psychophysical Predictions & Disagreement Analysis...")
    res_predictions = execute_all_predictions(zoo)

    # Compile unified summary table across all tasks
    summary_rows = []
    for m_id, obs in zoo.items():
        mf = res_modelfest["results"][m_id]
        wd = res_words["results"][m_id]
        bp = res_bandpass["results"][m_id]
        ph = res_phase["results"][m_id]
        bl = res_band_limited["results"][m_id]
        rd = res_rds["results"][m_id]
        at = res_attention["results"][m_id]
        wm = res_wm["results"][m_id]

        summary_rows.append({
            "model_id": m_id,
            "name": obs.name,
            "category": obs.category,
            "modelfest_r_human": mf["corr_with_human"],
            "modelfest_rmse": mf["rmse_log_cs"],
            "word_alpha_exponent": wd["alpha_word_length_exponent"],
            "face_critical_sf": bp["face_peak_sf"],
            "face_high_sf_texture_bias": bp["high_sf_bias_ratio"],
            "phase_sensitivity_index": ph["relative_phase_sensitivity_index"],
            "preserves_phase": ph["preserves_harmonic_phase"],
            "off_freq_masking_db": bl["low_pass_mask_db"],
            "cyclopean_stereopsis_pct": rd["stereoscopic_efficiency_pct"],
            "search_set_size_slope": at["set_size_log_log_slope"],
            "working_memory_k_capacity": wm["estimated_capacity_k_items"],
            "has_recurrent_memory": wm["has_recurrent_working_memory"],
        })

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_runtime_sec": round(time.perf_counter() - t0, 2),
        "modelfest": res_modelfest,
        "letter_word_inefficiency": res_words,
        "face_letter_bandpass": res_bandpass,
        "phase_reversal": res_phase,
        "band_limited_noise": res_band_limited,
        "random_dot_stereograms": res_rds,
        "visual_attention_search": res_attention,
        "working_memory": res_wm,
        "predictions": res_predictions,
        "summary_table": summary_rows,
    }

    data_dir = SCRIPT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "comprehensive_task_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    csv_path = data_dir / "comprehensive_task_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    js_dir = SCRIPT_DIR / "js"
    js_dir.mkdir(parents=True, exist_ok=True)
    js_path = js_dir / "comprehensive_task_data.js"
    with open(js_path, "w", encoding="utf-8") as f:
        f.write("// Auto-generated by vision_model_psychophysics/run_all_tasks.py\n")
        f.write("window.COMPREHENSIVE_PSYCHOPHYSICS_DATA = ")
        json.dump(payload, f, indent=2)
        f.write(";\n")

    print("\n" + "=" * 80)
    print(f" All 9 Psychophysical Tasks & Predictions Completed in {payload['total_runtime_sec']}s!")
    print(f"  - JSON Results: {json_path}")
    print(f"  - CSV Summary:  {csv_path}")
    print(f"  - JS Bundle:    {js_path}")
    print("=" * 80)
    return payload


if __name__ == "__main__":
    execute_comprehensive_suite()
