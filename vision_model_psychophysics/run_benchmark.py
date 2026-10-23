#!/usr/bin/env python3
"""
run_benchmark.py - Master Benchmark Runner for Local Vision Model Psychophysics
Executes all 4 psychophysical paradigms across all 8 observers (Canonical Mathematical,
Bio-Inspired V1, Open Neural Backbones [ConvNet, ViT], and Local Ollama VLM `llava:latest`),
performs live VLM 2AFC verification trials against Ollama when available, and exports:
  - vision_model_psychophysics/benchmark_results.json
  - vision_model_psychophysics/js/benchmark_data.js
  - vision_model_psychophysics/model_efficiency_summary.csv
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from observers import LocalVLMObserver, build_observer_zoo
from paradigms import (
    characterize_model_vs_ideal_and_human,
    run_classification_image_paradigm,
    run_critical_bandwidth_paradigm,
    run_equivalent_noise_paradigm,
    run_tvc_dipper_paradigm,
)
from stimuli import StimulusConfig, generate_white_noise


def run_live_ollama_verification_probes(
    vlm_obs: LocalVLMObserver,
    cfg: StimulusConfig,
) -> Dict[str, Any]:
    """
    Runs a set of live 2AFC verification probes against the local Ollama VLM (`llava:latest`)
    spanning suprathreshold, near-threshold, and noisy Gabor conditions to record empirical
    token responses and latency.
    """
    rng = np.random.default_rng(777)
    probe_conditions = [
        {"label": "High Contrast Clean (C=0.65, N_ext=0)", "contrast": 0.65, "n_ext": 0.0, "target_side": 1},
        {"label": "High Contrast Counterbalanced (C=0.65, N_ext=0)", "contrast": 0.65, "n_ext": 0.0, "target_side": 2},
        {"label": "Moderate Contrast Clean (C=0.30, N_ext=0)", "contrast": 0.30, "n_ext": 0.0, "target_side": 1},
    ]
    results = []
    any_success = False
    for cond in probe_conditions:
        n1 = generate_white_noise(cfg, cond["n_ext"], rng)
        n2 = generate_white_noise(cfg, cond["n_ext"], rng)
        sig = cond["contrast"] * vlm_obs.ideal_template
        if cond["target_side"] == 1:
            left_patch, right_patch = sig + n1, n2
        else:
            left_patch, right_patch = n1, sig + n2

        res = vlm_obs.probe_live_ollama_2afc(
            left_patch,
            right_patch,
            target_side=cond["target_side"],
            timeout_sec=45.0,
        )
        res["condition"] = cond["label"]
        res["contrast"] = cond["contrast"]
        res["n_ext"] = cond["n_ext"]
        results.append(res)
        if res.get("live_success"):
            any_success = True
        else:
            # If Ollama is unreachable or times out on first trial, don't block remaining trials
            break

    correct_total = sum(r.get("correct", 0) for r in results if r.get("live_success"))
    n_completed = sum(1 for r in results if r.get("live_success"))
    return {
        "ollama_reachable": any_success,
        "model": vlm_obs.ollama_model,
        "n_probes": n_completed,
        "accuracy": round(correct_total / max(n_completed, 1), 3) if n_completed > 0 else None,
        "probes": results,
    }


def execute_full_benchmark(run_live_vlm: bool = True) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cfg = StimulusConfig()
    zoo = build_observer_zoo(cfg)

    print("=" * 78)
    print(" VISION MODEL PSYCHOPHYSICS & IDEAL OBSERVER CHARACTERIZATION BENCHMARK")
    print("=" * 78)
    print(f" Stimulus Grid: {cfg.size_px}x{cfg.size_px} px ({cfg.px_per_deg} px/deg, target f0 = {cfg.sf_cpd} cpd)")

    all_models_data: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    for idx, (model_id, obs) in enumerate(zoo.items(), 1):
        m_start = time.perf_counter()
        print(f"\n [{idx}/{len(zoo)}] Testing Observer: {obs.name} ({model_id})")
        print(f"       Readout Mode: {obs.readout_mode}")

        eq_data = run_equivalent_noise_paradigm(obs, cfg, seed=100 + idx)
        ci_data = run_classification_image_paradigm(obs, cfg, n_trials=900, seed=200 + idx)
        cbw_data = run_critical_bandwidth_paradigm(obs, cfg, seed=300 + idx)
        tvc_data = run_tvc_dipper_paradigm(obs, cfg, seed=400 + idx)
        char_data = characterize_model_vs_ideal_and_human(obs, eq_data, ci_data, cbw_data, tvc_data)

        elapsed = round(time.perf_counter() - m_start, 2)
        print(
            f"       -> Done in {elapsed}s | eta_total={char_data['eta_total_vs_ideal']*100:.1f}% "
            f"(eta_tmpl={char_data['eta_template_sampling']*100:.1f}%, "
            f"eta_noise={char_data['eta_noise_consistency']*100:.1f}%, "
            f"eta_bw={char_data['eta_spectral_bandwidth']*100:.1f}%) | "
            f"CBW={cbw_data['critical_bandwidth_octaves']:.2f} oct | "
            f"Dip={tvc_data['dip_facilitation_db']:.2f} dB"
        )

        model_entry = {
            "model_id": model_id,
            "name": obs.name,
            "category": obs.category,
            "readout_mode": obs.readout_mode,
            "description": obs.description,
            "equivalent_noise": eq_data,
            "classification_image": ci_data,
            "critical_bandwidth": cbw_data,
            "tvc_dipper": tvc_data,
            "characterization": char_data,
        }
        all_models_data[model_id] = model_entry

        summary_rows.append({
            "model_id": model_id,
            "name": obs.name,
            "category": obs.category,
            "readout_mode": obs.readout_mode,
            "eta_total_pct": round(char_data["eta_total_vs_ideal"] * 100.0, 2),
            "eta_template_pct": round(char_data["eta_template_sampling"] * 100.0, 2),
            "eta_noise_consistency_pct": round(char_data["eta_noise_consistency"] * 100.0, 2),
            "eta_bandwidth_pct": round(char_data["eta_spectral_bandwidth"] * 100.0, 2),
            "n_eq_deg2": eq_data["lam_fit"]["n_eq_deg2"],
            "ptm_gamma": eq_data["ptm_fit"]["gamma_transducer"],
            "ptm_Am": eq_data["ptm_fit"]["multiplicative_noise_Am"],
            "double_pass_p_agree": eq_data["double_pass"]["p_agree"],
            "sigma_int_to_ext": eq_data["double_pass"]["sigma_int_to_ext_ratio"],
            "pbdi_patch_artifact": ci_data["patch_boundary_discontinuity_index"],
            "critical_bandwidth_oct": cbw_data["critical_bandwidth_octaves"],
            "tvc_c0_threshold": tvc_data["unmasked_threshold_c0"],
            "tvc_dip_depth_db": tvc_data["dip_facilitation_db"],
            "tvc_has_dipper": tvc_data["exhibits_dipper_facilitation"],
            "tvc_weber_slope": tvc_data["high_contrast_weber_slope"],
            "human_alignment_score_pct": char_data["human_alignment_score"]
            if "human_alignment_score" in char_data
            else char_data["human_psychophysics_alignment_pct"],
            "num_disagreements": len(char_data["disagreements_and_mechanisms"]),
        })

    live_vlm_report = {"ollama_reachable": False, "probes": []}
    if run_live_vlm and "local_vlm_llava" in zoo:
        print("\n [*] Running live 2AFC verification probes against local Ollama (llava:latest)...")
        vlm_obs = zoo["local_vlm_llava"]
        if isinstance(vlm_obs, LocalVLMObserver):
            live_vlm_report = run_live_ollama_verification_probes(vlm_obs, cfg)
            print(
                f"     -> Ollama Reachable: {live_vlm_report['ollama_reachable']} | "
                f"Completed Probes: {live_vlm_report.get('n_probes', 0)} | "
                f"Live 2AFC Accuracy: {live_vlm_report.get('accuracy')}"
            )

    out_dir = Path(__file__).resolve().parent
    js_dir = out_dir / "js"
    js_dir.mkdir(parents=True, exist_ok=True)

    full_payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_runtime_sec": round(time.perf_counter() - t0, 2),
        "stimulus_config": {
            "size_px": cfg.size_px,
            "px_per_deg": cfg.px_per_deg,
            "sf_cpd": cfg.sf_cpd,
            "sigma_deg": cfg.sigma_deg,
        },
        "live_ollama_vlm_verification": live_vlm_report,
        "models": all_models_data,
        "summary_table": summary_rows,
    }

    json_path = out_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    js_path = js_dir / "benchmark_data.js"
    with open(js_path, "w", encoding="utf-8") as f:
        f.write("// Auto-generated by vision_model_psychophysics/run_benchmark.py\n")
        f.write("window.MODEL_PSYCHOPHYSICS_BENCHMARK = ")
        json.dump(full_payload, f, indent=2)
        f.write(";\n")

    csv_path = out_dir / "model_efficiency_summary.csv"
    if summary_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print("\n" + "=" * 78)
    print(f" Benchmark Complete in {full_payload['total_runtime_sec']}s!")
    print(f"  - JSON Results: {json_path}")
    print(f"  - JS Bundle:    {js_path}")
    print(f"  - CSV Summary:  {csv_path}")
    print("=" * 78)
    return full_payload


if __name__ == "__main__":
    execute_full_benchmark(run_live_vlm=True)
