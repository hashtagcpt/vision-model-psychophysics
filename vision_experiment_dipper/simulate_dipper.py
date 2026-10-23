#!/usr/bin/env python3
"""
simulate_dipper.py - Psychophysical Simulation of Dipper Functions (TvC) with QUEST+
Simulates a synthetic human observer using the canonical Foley (1994) non-linear transducer
contrast discrimination model: R(C) = C^p / (C50^q + C^q).
"""

import math
import random
import sys

class SyntheticDipperObserver:
    """
    Exact Foley (1994) visual contrast transducer observer:
    R(C) = C^p / (C50^q + C^q)
    Detection/discrimination criterion: R(C_ped + Delta_C) - R(C_ped) = delta
    """
    def __init__(self):
        self.profiles = {
            ("ON", 0): {
                "name": "Center ON (0°)",
                "C50": 0.045,
                "p": 2.40,
                "q": 1.80,
                "delta": 0.011
            },
            ("OFF", 0): {
                "name": "Center OFF (0°)",
                "C50": 0.038,
                "p": 2.40,
                "q": 1.80,
                "delta": 0.008
            },
            ("ON", 4): {
                "name": "Periphery ON (4°)",
                "C50": 0.095,
                "p": 2.40,
                "q": 1.80,
                "delta": 0.020
            },
            ("OFF", 4): {
                "name": "Periphery OFF (4°)",
                "C50": 0.082,
                "p": 2.40,
                "q": 1.80,
                "delta": 0.015
            }
        }
        self.slope = 2.5
        self.gamma = 0.5  # 2IFC chance = 50%
        self.lapse = 0.02

    def _transducer_r(self, c, p_cfg):
        if c <= 0:
            return 0.0
        return math.pow(c, p_cfg["p"]) / (math.pow(p_cfg["C50"], p_cfg["q"]) + math.pow(c, p_cfg["q"]))

    def get_true_threshold(self, polarity, ecc_deg, c_ped):
        p_cfg = self.profiles[(polarity, ecc_deg)]
        r_ped = self._transducer_r(c_ped, p_cfg)
        r_target = r_ped + p_cfg["delta"]

        # Invert response via binary search to find exact Delta_C
        low, high = 1e-4, 1.0
        for _ in range(40):
            mid = (low + high) / 2.0
            if self._transducer_r(c_ped + mid, p_cfg) < r_target:
                low = mid
            else:
                high = mid
        return (low + high) / 2.0

    def respond(self, polarity, ecc_deg, c_ped, test_log_increment):
        true_inc = self.get_true_threshold(polarity, ecc_deg, c_ped)
        true_log_inc = math.log10(true_inc)
        x_diff = test_log_increment - true_log_inc
        # Weibull psychometric response probability
        p_correct = self.gamma + (1.0 - self.gamma - self.lapse) * (1.0 - math.exp(-math.pow(10.0, self.slope * x_diff)))
        p_correct = max(self.gamma, min(1.0 - self.lapse, p_correct))
        return 1 if random.random() < p_correct else 0


class QuestPlusPy:
    """Pure Python implementation of QuestPlus for simulation"""
    def __init__(self, t_min=-3.0, t_max=0.0, n_t=61, x_min=-2.9, x_max=0.0, n_x=51):
        self.nT = n_t
        self.nX = n_x
        self.thresholds = [t_min + i * (t_max - t_min) / (n_t - 1) for i in range(n_t)]
        self.intensities = [x_min + i * (x_max - x_min) / (n_x - 1) for i in range(n_x)]
        self.slope = 2.5
        self.gamma = 0.5
        self.lapse = 0.02

        self.posterior = [1.0 / self.nT] * self.nT
        self.history = []

        # Precompute likelihood
        self.lik_corr = []
        self.lik_inc = []
        for x in self.intensities:
            row_c = []
            row_i = []
            for t in self.thresholds:
                diff = x - t
                p = self.gamma + (1.0 - self.gamma - self.lapse) * (1.0 - math.exp(-math.pow(10.0, self.slope * diff)))
                p = max(1e-6, min(1.0 - 1e-6, p))
                row_c.append(p)
                row_i.append(1.0 - p)
            self.lik_corr.append(row_c)
            self.lik_inc.append(row_i)

    def get_optimal_stimulus(self):
        min_entropy = float("inf")
        best_x = self.intensities[0]

        for i, x in enumerate(self.intensities):
            p_corr = sum(self.lik_corr[i][j] * self.posterior[j] for j in range(self.nT))
            p_inc = 1.0 - p_corr

            h_corr = 0.0
            h_inc = 0.0
            for j in range(self.nT):
                p_tc = (self.lik_corr[i][j] * self.posterior[j]) / p_corr
                if p_tc > 1e-12:
                    h_corr -= p_tc * math.log2(p_tc)
                p_ti = (self.lik_inc[i][j] * self.posterior[j]) / p_inc
                if p_ti > 1e-12:
                    h_inc -= p_ti * math.log2(p_ti)

            exp_h = p_corr * h_corr + p_inc * h_inc
            if exp_h < min_entropy:
                min_entropy = exp_h
                best_x = x

        return best_x

    def update(self, test_x, response):
        idx = min(range(self.nX), key=lambda i: abs(self.intensities[i] - test_x))
        lik_row = self.lik_corr[idx] if response == 1 else self.lik_inc[idx]

        new_post = [lik_row[j] * self.posterior[j] for j in range(self.nT)]
        total = sum(new_post)
        if total > 0:
            self.posterior = [val / total for val in new_post]

    def get_threshold_mean(self):
        return sum(self.thresholds[j] * self.posterior[j] for j in range(self.nT))

    def get_threshold_std(self):
        mean = self.get_threshold_mean()
        var = sum(math.pow(self.thresholds[j] - mean, 2.0) * self.posterior[j] for j in range(self.nT))
        return math.sqrt(max(0.0, var))


def run_simulation():
    print("=" * 76)
    print(" VISION SCIENCE SIMULATION: Dipper Functions (TvC) for ON vs. OFF DoG")
    print(" Center (0 deg) and Periphery (4 deg) Retinal Eccentricity")
    print(" Canonical Foley (1994) Transducer Model: R(C) = C^p / (C50^q + C^q)")
    print("=" * 76)

    observer = SyntheticDipperObserver()
    polarities = ["ON", "OFF"]
    eccentricities = [0, 4]
    pedestals = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.50]
    trials_per_cond = 20

    # Build conditions
    conditions = []
    qp_dict = {}
    for pol in polarities:
        for ecc in eccentricities:
            for ped in pedestals:
                key = (pol, ecc, ped)
                conditions.append(key)
                qp_dict[key] = QuestPlusPy()

    total_trials = len(conditions) * trials_per_cond
    print(f" Simulating {len(conditions)} conditions x {trials_per_cond} trials = {total_trials} total trials...")

    # Interleaved simulation loop
    cond_pool = []
    for cond in conditions:
        cond_pool.extend([cond] * trials_per_cond)
    random.shuffle(cond_pool)

    correct_count = 0
    for trial_idx, cond in enumerate(cond_pool):
        pol, ecc, ped = cond
        qp = qp_dict[cond]
        test_x = qp.get_optimal_stimulus()
        resp = observer.respond(pol, ecc, ped, test_x)
        qp.update(test_x, resp)
        if resp == 1:
            correct_count += 1

    overall_acc = (correct_count / total_trials) * 100.0
    print(f" Simulation Complete! Overall 2IFC 2AFC Accuracy: {overall_acc:.1f}%\n")

    print("=" * 76)
    print(f"{'Condition':<20} | {'Pedestal':<9} | {'True dC':<8} | {'Est dC':<8} | {'True LogT':<9} | {'Est LogT':<8} | {'Dipper Stage'}")
    print("-" * 76)

    rmse_list = []
    dip_verified = 0

    for pol in polarities:
        for ecc in eccentricities:
            group_name = f"{pol} ({'Center 0°' if ecc==0 else 'Periph 4°'})"
            group_peds = [(pol, ecc, ped) for ped in pedestals]
            baseline_key = (pol, ecc, 0.0)
            baseline_est = math.pow(10.0, qp_dict[baseline_key].get_threshold_mean())

            has_dip = False
            for cond in group_peds:
                p, e, ped = cond
                qp = qp_dict[cond]
                true_dc = observer.get_true_threshold(p, e, ped)
                true_log = math.log10(true_dc)
                est_log = qp.get_threshold_mean()
                est_dc = math.pow(10.0, est_log)
                diff = est_log - true_log
                rmse_list.append(diff * diff)

                ped_str = f"{ped*100:.1f}%" if ped > 0 else "0% (Det)"
                is_fac = (est_dc < baseline_est * 0.85 and ped > 0)
                is_mask = (est_dc > baseline_est * 1.05 and ped > 0.15)
                
                if is_fac:
                    has_dip = True
                    stage_str = f"DIP ({baseline_est/est_dc:.1f}x fac)"
                elif is_mask:
                    stage_str = "MASKING (Weber)"
                elif ped == 0:
                    stage_str = "BASELINE (Det)"
                else:
                    stage_str = "TRANSITION"

                print(f"{group_name:<20} | {ped_str:<9} | {true_dc*100:6.2f}% | {est_dc*100:6.2f}% | {true_log:9.3f} | {est_log:8.3f} | {stage_str}")

            if has_dip:
                dip_verified += 1
            print("-" * 76)

    total_rmse = math.sqrt(sum(rmse_list) / len(rmse_list))
    print(f"Log10 Threshold Root-Mean-Square Error (RMSE): {total_rmse:.4f} log units")
    print(f"Facilitation Dips Verified across groups: {dip_verified}/4")

    assert total_rmse < 0.20, f"RMSE {total_rmse} exceeded threshold"
    assert dip_verified >= 3, f"Facilitation dip was not captured (only {dip_verified}/4)"
    print("\n[PASS] ALL PSYCHOPHYSICAL SIMULATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 76)

if __name__ == "__main__":
    run_simulation()
