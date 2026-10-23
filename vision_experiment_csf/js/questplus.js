/**
 * questplus.js - Full QUEST+ Bayesian Adaptive Psychometric Method (Watson, 2017)
 * 
 * Implements the multidimensional Bayesian adaptive algorithm for 2-Alternative
 * Forced Choice (2AFC) contrast threshold estimation across spatial frequencies.
 * Reference: Watson, A. B. (2017). QUEST+: A general multidimensional Bayesian
 * adaptive psychometric method. Journal of Vision, 17(3):10, 1-27.
 */

class QuestPlus {
    /**
     * @param {Object} options
     * @param {Array<number>} [options.thresholds] - Candidate threshold grid in log10 contrast (-3.0 to 0.0)
     * @param {Array<number>} [options.intensities] - Candidate test intensities in log10 contrast (-2.9 to 0.0)
     * @param {number} [options.slope=2.5] - Psychometric slope parameter (Weibull exponent)
     * @param {number} [options.guessRate=0.5] - Chance level (0.5 for 2-Alternative Forced Choice)
     * @param {number} [options.lapseRate=0.02] - Lapse / finger error rate at high contrast
     */
    constructor(options = {}) {
        // Candidate threshold space T (in log10 contrast: -3.0 to 0.0, e.g. 0.1% to 100% contrast)
        this.thresholds = options.thresholds || this._linspace(-3.0, 0.0, 61);
        
        // Candidate stimulus domain X to test
        this.intensities = options.intensities || this._linspace(-2.9, 0.0, 51);
        
        this.slope = options.slope !== undefined ? options.slope : 2.5;
        this.guessRate = options.guessRate !== undefined ? options.guessRate : 0.5; // 0.5 for 2AFC
        this.lapseRate = options.lapseRate !== undefined ? options.lapseRate : 0.02;

        this.nT = this.thresholds.length;
        this.nX = this.intensities.length;

        // Initialize uniform prior over threshold space T
        this.prior = new Float64Array(this.nT).fill(1.0 / this.nT);
        this.posterior = new Float64Array(this.prior);

        // Precompute likelihood matrix: P(correct | x_i, T_j)
        // Dimensions: [nX][nT]
        this.likelihoodCorrect = [];
        this.likelihoodIncorrect = [];
        this._precomputeLikelihoods();

        // History of trials for this QUEST+ instance
        this.history = [];
    }

    _linspace(start, end, n) {
        const step = (end - start) / (n - 1);
        const arr = [];
        for (let i = 0; i < n; i++) {
            arr.push(start + i * step);
        }
        return arr;
    }

    /**
     * Standard Weibull Psychometric function for log10 stimulus intensity
     * Psi(x; T, S) = gamma + (1 - gamma - lambda) * (1 - exp(-10^(S * (x - T))))
     */
    _psychometricFunction(x, T) {
        const exponent = Math.pow(10, this.slope * (x - T));
        const val = 1.0 - Math.exp(-exponent);
        return this.guessRate + (1.0 - this.guessRate - this.lapseRate) * val;
    }

    _precomputeLikelihoods() {
        for (let i = 0; i < this.nX; i++) {
            const x = this.intensities[i];
            const pCorr = new Float64Array(this.nT);
            const pIncorr = new Float64Array(this.nT);
            for (let j = 0; j < this.nT; j++) {
                const T = this.thresholds[j];
                const p = this._psychometricFunction(x, T);
                pCorr[j] = p;
                pIncorr[j] = 1.0 - p;
            }
            this.likelihoodCorrect.push(pCorr);
            this.likelihoodIncorrect.push(pIncorr);
        }
    }

    /**
     * Computes the Shannon entropy of a probability distribution
     */
    _entropy(pArr) {
        let h = 0.0;
        for (let i = 0; i < pArr.length; i++) {
            const p = pArr[i];
            if (p > 1e-12) {
                h -= p * Math.log2(p);
            }
        }
        return h;
    }

    /**
     * QUEST+ Minimum Expected Entropy Selection Rule
     * Evaluates all candidate stimulus intensities and selects x* that minimizes
     * the expected posterior entropy across outcomes (correct / incorrect).
     * @returns {number} optimal stimulus intensity x* in log10 contrast
     */
    getOptimalIntensity() {
        if (this.history.length === 0) {
            return this.getThresholdMean();
        }

        let bestIntensity = this.intensities[0];
        let minExpectedEntropy = Infinity;

        // Marginal probability of outcomes under current posterior:
        // P(r | x) = sum_j P(r | x, T_j) * p(T_j)
        for (let i = 0; i < this.nX; i++) {
            const pCorrGivenT = this.likelihoodCorrect[i];
            const pIncorrGivenT = this.likelihoodIncorrect[i];

            let pCorrectMarginal = 0.0;
            let pIncorrectMarginal = 0.0;

            for (let j = 0; j < this.nT; j++) {
                pCorrectMarginal += pCorrGivenT[j] * this.posterior[j];
                pIncorrectMarginal += pIncorrGivenT[j] * this.posterior[j];
            }

            // Simulated posterior if correct
            const postCorrect = new Float64Array(this.nT);
            const postIncorrect = new Float64Array(this.nT);

            if (pCorrectMarginal > 1e-12) {
                for (let j = 0; j < this.nT; j++) {
                    postCorrect[j] = (pCorrGivenT[j] * this.posterior[j]) / pCorrectMarginal;
                }
            }

            if (pIncorrectMarginal > 1e-12) {
                for (let j = 0; j < this.nT; j++) {
                    postIncorrect[j] = (pIncorrGivenT[j] * this.posterior[j]) / pIncorrectMarginal;
                }
            }

            const hCorrect = this._entropy(postCorrect);
            const hIncorrect = this._entropy(postIncorrect);

            const expectedEntropy = pCorrectMarginal * hCorrect + pIncorrectMarginal * hIncorrect;

            if (expectedEntropy < minExpectedEntropy) {
                minExpectedEntropy = expectedEntropy;
                bestIntensity = this.intensities[i];
            }
        }

        return bestIntensity;
    }

    /**
     * Bayesian Posterior Update:
     * p_{t+1}(T) = (P(response | x, T) * p_t(T)) / P(response | x)
     * @param {number} testIntensity - Log10 contrast tested
     * @param {boolean|number} isCorrect - 1 if correct, 0 if incorrect
     */
    update(testIntensity, isCorrect) {
        const correct = isCorrect ? 1 : 0;
        
        let closestIdx = 0;
        let minDiff = Infinity;
        for (let i = 0; i < this.nX; i++) {
            const diff = Math.abs(this.intensities[i] - testIntensity);
            if (diff < minDiff) {
                minDiff = diff;
                closestIdx = i;
            }
        }

        const likelihood = correct === 1 
            ? this.likelihoodCorrect[closestIdx] 
            : this.likelihoodIncorrect[closestIdx];

        let totalProb = 0.0;
        for (let j = 0; j < this.nT; j++) {
            this.posterior[j] *= likelihood[j];
            totalProb += this.posterior[j];
        }

        if (totalProb > 1e-15) {
            for (let j = 0; j < this.nT; j++) {
                this.posterior[j] /= totalProb;
            }
        } else {
            this.posterior.fill(1.0 / this.nT);
        }

        const currentMean = this.getThresholdMean();
        const currentEntropy = this.getCurrentEntropy();

        this.history.push({
            trial: this.history.length + 1,
            intensity: testIntensity,
            contrast: Math.pow(10, testIntensity),
            correct: correct,
            estimatedThreshold: currentMean,
            estimatedContrastThreshold: Math.pow(10, currentMean),
            entropy: currentEntropy
        });
    }

    getThresholdMean() {
        let mean = 0.0;
        for (let j = 0; j < this.nT; j++) {
            mean += this.thresholds[j] * this.posterior[j];
        }
        return mean;
    }

    getThresholdMode() {
        let maxP = -1;
        let mode = this.thresholds[0];
        for (let j = 0; j < this.nT; j++) {
            if (this.posterior[j] > maxP) {
                maxP = this.posterior[j];
                mode = this.thresholds[j];
            }
        }
        return mode;
    }

    getThresholdStd() {
        const mean = this.getThresholdMean();
        let variance = 0.0;
        for (let j = 0; j < this.nT; j++) {
            const diff = this.thresholds[j] - mean;
            variance += diff * diff * this.posterior[j];
        }
        return Math.sqrt(variance);
    }

    getCurrentEntropy() {
        return this._entropy(this.posterior);
    }

    getEntropy() {
        return this.getCurrentEntropy();
    }

    getOptimalStimulus() {
        return this.getOptimalIntensity();
    }

    getCredibleInterval(level = 0.95) {
        const tail = (1.0 - level) / 2.0;
        let cum = 0.0;
        let lower = this.thresholds[0];
        let upper = this.thresholds[this.nT - 1];
        let foundLower = false;

        for (let j = 0; j < this.nT; j++) {
            cum += this.posterior[j];
            if (!foundLower && cum >= tail) {
                lower = this.thresholds[j];
                foundLower = true;
            }
            if (cum >= (1.0 - tail)) {
                upper = this.thresholds[j];
                break;
            }
        }
        return { lower, upper };
    }

    getLinearContrastThreshold() {
        return Math.pow(10, this.getThresholdMean());
    }

    getContrastSensitivity() {
        const lin = this.getLinearContrastThreshold();
        return 1.0 / Math.max(1e-5, lin);
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { QuestPlus };
}
