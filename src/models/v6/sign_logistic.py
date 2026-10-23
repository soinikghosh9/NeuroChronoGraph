"""
Sign-constrained logistic regression: clinical directions guaranteed, output continuous.

Motivation from measurement, not preference. The monotone-constrained boosted head gave the
right clinical directions but was substantially less reliable than a random forest
(split-half ICC 0.846 vs 0.998, paired -0.153, p<0.001). The mechanism is that a boosted tree is
a **step function**: a small shift in a concept median crosses a split threshold and moves the
output discontinuously, and a single pass through 300 trees has no averaging to absorb it.

A logistic model is continuous in its inputs, so the score moves smoothly with the concepts. The
question this class answers is whether continuity recovers the stability while keeping the
clinical-direction guarantee.

**How the constraint is enforced.** Features whose clinical direction is -1 are sign-flipped, so
every constrained feature is expected to have a POSITIVE coefficient. The fit then minimises the
L2-regularised logistic loss subject to ``w >= 0`` on those coordinates, via bounded L-BFGS-B.
Unconstrained concepts (direction 0) and the intercept are left free. The result is exact -- not
a penalty that discourages wrong signs, but a feasible region that excludes them.

scikit-learn has no coefficient-sign constraint, hence the explicit optimiser.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize


class SignConstrainedLogistic:
    """Logistic regression with per-coefficient sign constraints.

    Parameters
    ----------
    directions
        One of -1, 0, +1 per feature. -1 means the feature DECREASES with the positive class,
        +1 increases, 0 leaves it free.
    C
        Inverse L2 strength, as in scikit-learn.
    class_weight
        ``"balanced"`` reweights by inverse class frequency.
    """

    def __init__(self, directions: Sequence[int], C: float = 1.0,
                 class_weight: Optional[str] = "balanced", max_iter: int = 2000):
        self.directions = np.asarray(directions, dtype=float)
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape

        # standardise, then flip so every constrained coefficient should be >= 0
        self.mu_ = X.mean(0)
        self.sd_ = X.std(0) + 1e-9
        self.flip_ = np.where(self.directions < 0, -1.0, 1.0)
        Z = ((X - self.mu_) / self.sd_) * self.flip_

        if self.class_weight == "balanced":
            w_pos = n / (2.0 * max(y.sum(), 1))
            w_neg = n / (2.0 * max((1 - y).sum(), 1))
            sw = np.where(y > 0.5, w_pos, w_neg)
        else:
            sw = np.ones(n)
        sw = sw / sw.mean()

        lam = 1.0 / (2.0 * self.C * n)

        def obj(theta):
            w, b = theta[:d], theta[d]
            z = Z @ w + b
            # stable log(1+exp(z))
            log1pexp = np.logaddexp(0.0, z)
            nll = np.sum(sw * (log1pexp - y * z)) / n
            p = 1.0 / (1.0 + np.exp(-z))
            g = Z.T @ (sw * (p - y)) / n + 2.0 * lam * w
            gb = float(np.sum(sw * (p - y)) / n)
            return nll + lam * float(w @ w), np.concatenate([g, [gb]])

        # constrained coordinates are bounded below at zero AFTER the flip
        bounds = [((0.0, None) if dvec != 0 else (None, None)) for dvec in self.directions]
        bounds.append((None, None))                       # intercept

        res = minimize(obj, np.zeros(d + 1), jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": self.max_iter})
        self.coef_flipped_ = res.x[:d]
        self.intercept_ = float(res.x[d])
        # coefficients in the ORIGINAL feature orientation, for reporting
        self.coef_ = self.coef_flipped_ * self.flip_ / self.sd_
        self.coef_standardised_ = self.coef_flipped_ * self.flip_
        self.success_ = bool(res.success)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        Z = ((X - self.mu_) / self.sd_) * self.flip_
        return Z @ self.coef_flipped_ + self.intercept_

    def predict_proba(self, X):
        z = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p, p])

    def contributions(self, X):
        """Per-feature contribution to the log-odds, for the clinical report.

        Exact by construction: the score IS the sum of these plus the intercept, so a report
        built from them is not an approximation of the model, it is the model.
        """
        X = np.asarray(X, dtype=float)
        Z = ((X - self.mu_) / self.sd_) * self.flip_
        return Z * self.coef_flipped_
