"""
Per-electrode spectral descriptors for NeuroChronoGraph v5.

Computed in torch, inside the model, on whatever window length it is given. That is
deliberate: it needs no dataloader change, no feature cache to invalidate when the window
length changes, and it runs on GPU alongside everything else.

Every feature here is **gain-invariant by construction** -- relative powers, ratios,
frequencies and slopes. None of them changes if the whole recording is multiplied by a
constant. This is the analytic alternative to domain randomisation, which failed in this
project (fold 0 kappa 0.487 -> 0.335) precisely because amplifier gain and the disease
signal share the frequency axis, so any randomisation wide enough to cover the nuisance
also destroyed the signal.

The 12 features per electrode:

    0-4   relative band power (delta, theta, alpha, beta, gamma)
    5     individual alpha peak frequency (IAPF), parabolically interpolated
    6     alpha peak prominence (log ratio of peak to local median)
    7     spectral edge frequency (95% of cumulative power)
    8     aperiodic 1/f exponent (log-log slope, alpha band excluded from the fit)
    9     theta / alpha ratio (log1p)
    10    (delta+theta) / (alpha+beta) ratio (log1p)
    11    spectral entropy (normalised to [0, 1])

Features 5, 6 and 8 are not computed anywhere else in this repository. IAPF is the
best-replicated marker in the quantitative EEG dementia literature and the v2 architecture
had no mechanism capable of representing it: two MaxPool1d(4) give a 12.5 Hz effective
feature rate, so oscillations above ~6 Hz survive only as slowly varying envelopes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

BANDS: Tuple[Tuple[float, float], ...] = (
    (0.5, 4.0),    # delta
    (4.0, 8.0),    # theta
    (8.0, 13.0),   # alpha
    (13.0, 30.0),  # beta
    (30.0, 45.0),  # gamma
)

BAND_NAMES = ("delta", "theta", "alpha", "beta", "gamma")


class SpectralNodeFeatures(nn.Module):
    """Map a raw window ``[B, C, T]`` to a per-electrode descriptor ``[B, C, 12]``.

    Parameters
    ----------
    sfreq
        Sampling rate in Hz.
    seg_seconds
        Welch segment length. 2 s gives 0.5 Hz resolution.
    alpha_search
        Range searched for the alpha peak. Wider than the alpha band itself (8-13 Hz)
        because the peak *slows* in dementia -- searching only 8-13 would clip exactly the
        pathological cases this is meant to detect.
    fit_range
        Frequency range for the aperiodic log-log fit.
    exclude_from_fit
        Band excluded from the aperiodic fit so the alpha peak does not bias the slope.
    entropy_band
        Range over which spectral entropy is computed. ``None`` keeps the historical
        full-spectrum behaviour. Restricting it matters because electromyographic activity is
        broadband and dominates above roughly 20 Hz, so a full-spectrum entropy is partly a
        measure of muscle tone; measured on this corpus the full-spectrum version correlates
        +0.708 with relative gamma power, more strongly than it correlates with MMSE.

    Note on defaults: these reproduce the v5 feature set exactly and must not be changed, since
    trained v5 checkpoints depend on them. The v6 concept extractor overrides them explicitly.
    """

    N_FEATURES = 12

    def __init__(self,
                 sfreq: float = 256.0,
                 seg_seconds: float = 2.0,
                 alpha_search: Tuple[float, float] = (6.0, 14.0),
                 fit_range: Tuple[float, float] = (2.0, 40.0),
                 exclude_from_fit: Tuple[float, float] = (7.0, 14.0),
                 edge_fraction: float = 0.95,
                 edge_band: Tuple[float, float] = (1.0, 30.0),
                 entropy_band: Optional[Tuple[float, float]] = None,
                 normalise: bool = True):
        super().__init__()
        self.sfreq = float(sfreq)
        self.nperseg = max(int(seg_seconds * sfreq), 64)
        self.alpha_search = alpha_search
        self.fit_range = fit_range
        self.exclude_from_fit = exclude_from_fit
        self.edge_fraction = edge_fraction
        # Spectral edge is computed over 1-30 Hz, NOT the full spectrum. Measured on this
        # corpus, a full-spectrum edge correlates r=+0.816 with relative gamma power -- it was
        # tracking EMG contamination rather than slowing, and consequently carried a
        # clinically inverted weight (higher edge -> more impaired, when slowing should lower
        # it). Restricting to the clinical analysis band removes the muscle-artefact pathway.
        self.edge_band = edge_band
        self.entropy_band = entropy_band
        self.normalise = normalise
        self.eps = 1e-10
        self.register_buffer("_hann", torch.hann_window(self.nperseg), persistent=False)

    # ------------------------------------------------------------------ spectrum
    def _welch(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Welch PSD. Returns ``freqs [F]`` and ``psd [B, C, F]``."""
        B, C, T = x.shape
        nperseg = min(self.nperseg, T)
        hop = max(nperseg // 2, 1)

        if T >= nperseg:
            seg = x.unfold(-1, nperseg, hop)            # [B, C, nseg, nperseg]
        else:                                            # pad short windows
            seg = torch.nn.functional.pad(x, (0, nperseg - T)).unsqueeze(2)

        win = self._hann[:nperseg] if nperseg == self.nperseg else torch.hann_window(
            nperseg, device=x.device, dtype=x.dtype)
        seg = seg - seg.mean(dim=-1, keepdim=True)       # detrend (removes DC)
        seg = seg * win

        spec = torch.fft.rfft(seg, dim=-1)
        scale = 1.0 / (self.sfreq * (win.pow(2).sum() + self.eps))
        psd = (spec.real.pow(2) + spec.imag.pow(2)) * scale
        psd = psd.mean(dim=2)                            # average segments -> [B, C, F]

        freqs = torch.fft.rfftfreq(nperseg, d=1.0 / self.sfreq).to(x.device)
        return freqs, psd

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _band_mask(freqs: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return (freqs >= lo) & (freqs < hi)

    def _alpha_peak(self, freqs: torch.Tensor, logpsd: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parabolically-interpolated peak frequency and prominence in the search range."""
        lo, hi = self.alpha_search
        mask = self._band_mask(freqs, lo, hi)
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if idx.numel() < 3:
            # F3: previously returned a hard-coded 10.0 Hz -- a value indistinguishable from a
            # healthy posterior rhythm, emitted precisely when the estimate is undefined. NaN
            # propagates to the subject aggregate, where it is counted and flagged rather than
            # silently reported as a normal finding.
            B, C, _ = logpsd.shape
            nan = torch.full((B, C), float("nan"), device=logpsd.device, dtype=logpsd.dtype)
            return nan, nan.clone()

        sub = logpsd[..., idx]                                    # [B, C, K]
        k = sub.argmax(dim=-1)                                    # [B, C]

        # Parabolic interpolation on the log spectrum: shifts the estimate off the FFT
        # grid, which matters because a 0.5 Hz bin is coarse next to the ~1-2 Hz slowing
        # this is meant to resolve.
        kc = k.clamp(1, sub.shape[-1] - 2)
        y0 = sub.gather(-1, (kc - 1).unsqueeze(-1)).squeeze(-1)
        y1 = sub.gather(-1, kc.unsqueeze(-1)).squeeze(-1)
        y2 = sub.gather(-1, (kc + 1).unsqueeze(-1)).squeeze(-1)
        denom = (y0 - 2 * y1 + y2)
        offset = torch.where(denom.abs() > 1e-9, 0.5 * (y0 - y2) / denom,
                             torch.zeros_like(denom)).clamp(-1.0, 1.0)

        df = (freqs[1] - freqs[0])
        peak_f = freqs[idx][kc] + offset * df
        # Prominence: how far the peak stands above the typical level in the search band.
        prominence = y1 - sub.median(dim=-1).values
        return peak_f, prominence

    def _aperiodic_exponent(self, freqs: torch.Tensor, logpsd: torch.Tensor) -> torch.Tensor:
        """OLS slope of log10(PSD) against log10(f); returned as a positive exponent."""
        lo, hi = self.fit_range
        ex_lo, ex_hi = self.exclude_from_fit
        mask = self._band_mask(freqs, lo, hi) & ~self._band_mask(freqs, ex_lo, ex_hi)
        mask = mask & (freqs > 0)
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if idx.numel() < 3:
            return torch.zeros(logpsd.shape[:2], device=logpsd.device, dtype=logpsd.dtype)

        lf = torch.log10(freqs[idx])                     # [K]
        lf = lf - lf.mean()
        y = logpsd[..., idx] / 2.302585092994046         # ln -> log10
        y = y - y.mean(dim=-1, keepdim=True)
        slope = (y * lf).sum(dim=-1) / (lf.pow(2).sum() + self.eps)
        return -slope                                    # positive for 1/f^beta

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def absolute_band_power(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, C, T]`` -> ``[B, C, 5]`` band power WITHOUT within-channel normalisation.

        Everything `forward` returns is divided by that channel's total power and is therefore
        gain-invariant by construction. That is the right choice for the features the model
        consumes, and it is exactly why three concepts named after inter-electrode *amplitude*
        statements cannot express them: a within-channel ratio cannot see how large one electrode
        is relative to another.

        This method is the missing ingredient. It is deliberately not wired into `forward`, so no
        existing feature changes; it exists to build amplitude-sensitive concepts alongside the
        relative ones. Note that these powers are only meaningful when the upstream scaling
        preserves inter-channel amplitude -- i.e. under global scaling, not per-channel scaling.
        """
        x = x.detach().float()
        freqs, psd = self._welch(x)
        return torch.stack([psd[..., self._band_mask(freqs, lo, hi)].sum(dim=-1)
                            for lo, hi in BANDS], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, C, T]`` raw window. Returns ``[B, C, 12]``."""
        x = x.detach().float()                           # features, not a gradient path
        freqs, psd = self._welch(x)
        logpsd = torch.log(psd + self.eps)

        total = psd.sum(dim=-1, keepdim=True) + self.eps
        rel = []
        for lo, hi in BANDS:
            m = self._band_mask(freqs, lo, hi)
            rel.append(psd[..., m].sum(dim=-1) / total.squeeze(-1))
        rel_t = torch.stack(rel, dim=-1)                 # [B, C, 5]

        peak_f, prominence = self._alpha_peak(freqs, logpsd)
        exponent = self._aperiodic_exponent(freqs, logpsd)

        # Spectral edge frequency: lowest f holding `edge_fraction` of the IN-BAND power.
        eb = self._band_mask(freqs, *self.edge_band)
        eb_idx = torch.nonzero(eb, as_tuple=False).squeeze(-1)
        if eb_idx.numel() >= 2:
            psd_b = psd[..., eb_idx]
            cum_b = psd_b.cumsum(dim=-1) / (psd_b.sum(dim=-1, keepdim=True) + self.eps)
            off = (cum_b < self.edge_fraction).sum(dim=-1).clamp(max=eb_idx.numel() - 1)
            edge_f = freqs[eb_idx][off]
        else:
            edge_f = torch.full(psd.shape[:2], float(self.edge_band[1]),
                                device=x.device, dtype=psd.dtype)

        delta, theta, alpha, beta = rel_t[..., 0], rel_t[..., 1], rel_t[..., 2], rel_t[..., 3]
        tar = torch.log1p(theta / (alpha + self.eps))
        slowing = torch.log1p((delta + theta) / (alpha + beta + self.eps))

        # Spectral entropy. Renormalised WITHIN the chosen band, so it measures how flat the
        # spectrum is across that band rather than how much power sits outside it.
        if self.entropy_band is None:
            psd_e = psd
        else:
            em = self._band_mask(freqs, *self.entropy_band)
            psd_e = psd[..., em] if em.sum() >= 2 else psd
        pn = psd_e / (psd_e.sum(dim=-1, keepdim=True) + self.eps)
        ent = -(pn * torch.log(pn + self.eps)).sum(dim=-1) / torch.log(
            torch.tensor(float(pn.shape[-1]), device=x.device))

        if self.normalise:
            # Put everything on a comparable O(1) scale with fixed physiological constants,
            # not batch statistics -- batch norms would leak cohort information across a
            # batch, which is exactly the shortcut this project is trying to remove.
            peak_f = (peak_f - 10.0) / 3.0
            prominence = prominence / 3.0
            edge_f = (edge_f - 20.0) / 10.0
            exponent = (exponent - 1.5)
            tar = tar / 2.0
            slowing = slowing / 2.0

        return torch.stack(
            [rel_t[..., 0], rel_t[..., 1], rel_t[..., 2], rel_t[..., 3], rel_t[..., 4],
             peak_f, prominence, edge_f, exponent, tar, slowing, ent],
            dim=-1,
        )
