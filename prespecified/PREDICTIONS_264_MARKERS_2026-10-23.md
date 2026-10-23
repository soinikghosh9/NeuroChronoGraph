# Fixed predictions for experiment 264 (marker validation), written 2026-10-23 before it was run

Status: written before `experiments/264_marker_validation.py` existed or was run. Its SHA-256 is recorded in
`STATUS_REVISION_2026-10-12.md` at the time of writing. Every analysis below is reported whatever it shows. No
analysis here chooses a model, feature, or threshold; SpectraLite and its calibration rule are unchanged.

## What was already seen (exploratory audit, 2026-10-13; `canonical/marker_audit_explore.json`)
- Whole-head AD-vs-CN Hedges g point estimates at CAUEEG, ds004504, P-ADIC, and BrainLat (9 of 12 markers had
  the same sign at all four).
- Healthy-control shifts against CAUEEG controls (whole-head point estimates).
- Split-half reliability within CAUEEG only.
- MCI-vs-CN whole-head point estimates (small, mixed signs).

These are confirmatory restatements with intervals, not tests: predictions C1 to C3 are marked as such.

## Not yet computed (new predictions)
- Intervals, random-effects pooling, and I2.
- Regional effects.
- Split-half reliability at the external cohorts.
- Cross-session agreement (eyes-closed versus photic) in the same subjects.
- Association with MMSE or MoCA.
- Any link between a control's markers and its SpectraLite score or false referral.
- Marker-level threshold transport.

## Definitions (fixed now)
- **Marker.** One of SpectraLite's 12 node features. Per subject: the median over its 24 windows, then the mean over
  the available electrodes (whole head) or over a region's electrodes. Regions: frontal (Fp1, Fp2, F3, F4, F7, F8,
  Fz), central (C3, C4, Cz), temporal (T3, T4, T5, T6), parietal (P3, P4, Pz), occipital (O1, O2).
- **Subjects.** Those scored in the corresponding SpectraLite task.
  - CN_vs_dementia score files: CAUEEG out-of-fold, ds004504, P-ADIC, BrainLat. AD-vs-CN uses AD and CN only.
  - MCI cohorts: CAUEEG, GENEEG, P-ADIC, MISP, using the subjects of the CN-vs-MCI task.
- **Effect.** Hedges g (case minus control) with a 2,000-resample subject bootstrap 95% CI. Pooled across cohorts by
  DerSimonian-Laird random effects, with I2.
- **Healthy-baseline shift.** g of a cohort's controls against CAUEEG controls.
- **Disease axis** (fitted on CAUEEG only):
  - an L2 logistic regression (C = 1) on the 12 whole-head markers, standardized by CAUEEG controls, CN versus dementia;
  - the axis value of any subject is its linear predictor.
- **Transferred threshold.** The 80th percentile of SpectraLite's CAUEEG out-of-fold control scores, as in the paper.
  A control is falsely referred if its score is at or above it.
- **Reliability.**
  - Split-half: Pearson r between the medians of windows 1 to 12 and 13 to 24, Spearman-Brown corrected.
  - Cross-session: ICC(3,1) between the eyes-closed (ds004504) and photic (ds006036) whole-head markers of the
    same 69 subjects.

## Predictions
- **C1 (confirmatory).** Pooled AD-vs-CN g is positive for relative theta, theta/alpha, and slowing, and negative
  for IAF, each with a 95% CI excluding zero.
- **C2 (confirmatory).** For at least 6 of the 12 markers, at least one external cohort's controls differ from
  CAUEEG controls by |g| >= 0.5.
- **C3 (confirmatory).** No marker has a pooled MCI-vs-CN g with |g| >= 0.3 and a CI excluding zero.
- **N1.** The median split-half reliability over the 9 sign-consistent markers is at least 0.80 at every cohort.
- **N2.** The median cross-session ICC(3,1) over the same 9 markers is at least 0.60.
- **N3.** Posterior (parietal and occipital) theta/alpha g exceeds frontal theta/alpha g at 3 or more of the 4
  dementia cohorts.
- **N4.** Within AD cases, theta/alpha and slowing correlate negatively, and IAF positively, with MMSE. Tested at
  ds004504 and P-ADIC pooled (Fisher z, cohort-stratified); BrainLat MoCA is reported separately.
- **N5 (mechanism, primary).** Among the controls of the three external dementia cohorts pooled:
  - a control's disease-axis value correlates positively with its SpectraLite score (Spearman rho > 0, 95%
    bootstrap CI excluding 0);
  - falsely referred controls have higher axis values than other controls (AUC > 0.5, CI excluding 0.5).
- **N6 (mechanism, cohort level; descriptive, n = 3).** The order of the three cohorts' mean control axis shift
  (against CAUEEG controls) is the reverse of the order of their transferred-threshold specificity.
- **N7.** A single-marker cutoff transferred from CAUEEG controls (the 80th percentile of theta/alpha) gives external
  specificity below 0.80 at 2 or more of the 3 dementia cohorts. The same cutoff set on 20 local controls
  (split-conformal rank, 200 splits) gives mean specificity within 0.05 of 0.80. The latter is expected under
  exchangeability; it is reported as a check.
- **N8 (descriptive, n = 12 markers).** Integrated-gradients share (attribution12.json) correlates positively with
  |pooled AD g| (Spearman).

## Reporting
- Every prediction is reported as held, not held, or not testable, with its estimate and interval.
- Failures are reported in the paper.
- Nothing is rerun with changed definitions after seeing results. Any later change is labelled post hoc here, with
  its date.
