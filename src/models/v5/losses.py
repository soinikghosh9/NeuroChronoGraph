"""
Split-half consistency: a differentiable surrogate for test-retest reliability.

Motivation, from this project's own measurements. Test-retest ICC(2,1) for the screening
score is **0.342**, and **31% of screening decisions flip** when the same patient is
recorded again. A tool that changes its answer for roughly one patient in three cannot be
deployed at any AUC, which makes reliability -- not accuracy -- the binding constraint.

Experiment L1 established that no *aggregation* rule fixes this: logit-mean, median,
trimmed mean and fixed-K sampling all landed in the same place. That is the diagnostic
result. The instability is upstream of aggregation, in the per-window posterior itself, so
it has to be addressed in the training objective rather than the read-out.

This loss splits a subject's windows into two disjoint halves and penalises disagreement
between the two subject-level posteriors. Within-session split-half is a lower bound on
between-session variation -- it cannot capture electrode replacement or day-to-day state --
but it is differentiable, needs no repeat recordings, and targets the right quantity.

**The failure mode is real and must be monitored.** A constant output minimises this loss
perfectly. Every guard below exists for that reason:

* ``warmup_epochs`` -- the classifier converges before consistency is applied at all;
* a small weight (``lambda`` ~ 0.1), so it regularises rather than dominates;
* :func:`consistency_diagnostics` reports the spread of predictions across subjects, which
  goes to zero exactly when the model is collapsing. **If spread falls while consistency
  improves, the term is winning for the wrong reason and the run should be stopped.**
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_indices(subject_ids: Sequence) -> Dict[object, torch.Tensor]:
    groups: Dict[object, list] = {}
    for i, s in enumerate(subject_ids):
        key = s.item() if torch.is_tensor(s) else s
        groups.setdefault(key, []).append(i)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in groups.items()}


def split_half_posteriors(probs: torch.Tensor,
                          subject_ids: Sequence,
                          generator: Optional[torch.Generator] = None,
                          min_windows: int = 4
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split each subject's windows in two and mean-pool each half.

    ``probs`` is ``[N, n_classes]`` over windows; ``subject_ids`` has length ``N``.
    Subjects with fewer than ``min_windows`` windows are skipped -- a two-window split is
    pure noise and would teach the model to be consistent about nothing.

    Returns two tensors ``[n_subjects_used, n_classes]``.
    """
    halves_a, halves_b = [], []
    for _, idx in _group_indices(subject_ids).items():
        if idx.numel() < min_windows:
            continue
        perm = torch.randperm(idx.numel(), generator=generator, device='cpu')
        idx = idx[perm].to(probs.device)
        cut = idx.numel() // 2
        halves_a.append(probs[idx[:cut]].mean(dim=0))
        halves_b.append(probs[idx[cut:2 * cut]].mean(dim=0))

    if not halves_a:
        empty = probs.new_zeros((0, probs.shape[-1]))
        return empty, empty
    return torch.stack(halves_a), torch.stack(halves_b)


class SplitHalfConsistencyLoss(nn.Module):
    """Penalise disagreement between two disjoint halves of a subject's windows.

    Parameters
    ----------
    weight
        Final lambda. Deliberately small; this is a regulariser, not an objective.
    warmup_epochs
        Epochs during which the weight is zero. Applying consistency before the classifier
        has learned anything is the fastest route to the constant solution.
    ramp_epochs
        Epochs over which the weight rises linearly from 0 to ``weight`` after warmup.
    mode
        ``'l2'`` on the posteriors, or ``'sym_kl'`` symmetric KL (scale-free on the simplex,
        but sharper gradients near confident predictions).
    """

    def __init__(self,
                 weight: float = 0.1,
                 warmup_epochs: int = 5,
                 ramp_epochs: int = 5,
                 mode: str = "l2",
                 min_windows: int = 4):
        super().__init__()
        if mode not in ("l2", "sym_kl"):
            raise ValueError(f"mode must be 'l2' or 'sym_kl', got {mode!r}")
        self.weight = weight
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = max(ramp_epochs, 1)
        self.mode = mode
        self.min_windows = min_windows

    def current_weight(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return 0.0
        progress = (epoch - self.warmup_epochs) / self.ramp_epochs
        return self.weight * min(progress, 1.0)

    def forward(self,
                probs: torch.Tensor,
                subject_ids: Sequence,
                epoch: int = 10**6,
                generator: Optional[torch.Generator] = None
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        w = self.current_weight(epoch)
        a, b = split_half_posteriors(probs, subject_ids, generator, self.min_windows)

        if a.shape[0] == 0:
            zero = probs.sum() * 0.0
            return zero, {"consistency_raw": 0.0, "consistency_weight": w, "n_subjects": 0}

        if self.mode == "l2":
            raw = (a - b).pow(2).sum(dim=-1).mean()
        else:
            la, lb = torch.log(a + 1e-8), torch.log(b + 1e-8)
            raw = 0.5 * (F.kl_div(lb, a, reduction="batchmean")
                         + F.kl_div(la, b, reduction="batchmean"))

        stats = {
            "consistency_raw": float(raw.detach()),
            "consistency_weight": w,
            "n_subjects": int(a.shape[0]),
        }
        stats.update(consistency_diagnostics(a, b))
        return w * raw, stats


@torch.no_grad()
def consistency_diagnostics(a: torch.Tensor, b: torch.Tensor,
                            normal_class: int = 2) -> Dict[str, float]:
    """Collapse detector and a direct analogue of the measured decision-flip rate.

    ``pred_spread`` is the std of the impairment score across subjects. **It going to zero
    is what collapse looks like** -- the model has found the constant that minimises the
    consistency term. Watch it against discrimination on every run.

    ``flip_rate`` is the fraction of subjects whose argmax differs between the two halves,
    the within-session analogue of the 31% between-session flip rate that motivated this.
    """
    if a.shape[0] == 0:
        return {"pred_spread": 0.0, "flip_rate": 0.0}
    # Impairment score = 1 - P(normal). NOTE the normal class is index 2 in this project's
    # four-class scheme (0=AD/dementia, 1=FTD, 2=CN, 3=MCI), not index 0 -- see the OOF
    # records in outputs/results/clean_protocol/.
    c = min(normal_class, a.shape[-1] - 1)
    score = 1.0 - 0.5 * (a[:, c] + b[:, c])
    flips = (a.argmax(dim=-1) != b.argmax(dim=-1)).float().mean()
    return {
        "pred_spread": float(score.std()) if score.numel() > 1 else 0.0,
        "flip_rate": float(flips),
    }
