# Real-time causal detection of freezing of gait: technical report

Zeynep Kocacenk. Research prototype; no clinical claims. All numbers below are produced by the code in
this repository from saved result files (`results/`), which the tests and `make_report.py` read.

## 1. Problem

A detector that cues a person with Parkinson's disease when they freeze, or alerts a clinician during
a procedure, has to decide at time *t* from samples up to *t*, within a frame budget, for a person it
was not trained on. Offline evaluation (whole recording visible, random frame splits, thresholds tuned
on the test data) answers a different question. This report evaluates causal sequence models under
the deployed conditions and is explicit about what 8 test patients can and cannot show.

## 2. Data and protocol

* **Daphnet Freezing of Gait** (Bachlin et al., 2010; UCI, CC BY 4.0). 10 patients, three 3-axis
  accelerometers (ankle, thigh, trunk) at 64 Hz, 17 recordings. Frames annotated "not part of the
  experiment" are dropped and recordings split at those gaps (35 segments). Pairs of samples are
  averaged (causal) to 32 Hz: a 31.25 ms frame budget. Labels: freeze vs. no freeze.
* **Leave one patient out with validation patients.** The 8 patients with at least one freeze are
  test patients in turn (S04 and S10 never freeze; they are only used for training). In each fold two
  other freezing patients, drawn by seed, are validation patients; the remaining patients train.
  Normalisation statistics come from the training patients. **Every choice** (epoch by validation
  PR-AUC, decision threshold, debounce, Freeze Index thresholds, temperature, abstention level) uses
  validation patients only. 3 seeds change the validation draw, initialisation and data order.
* **Leakage checks** (`tests/test_daphnet.py`, review in section 9): subjects never overlap between
  train, validation and test; normalisation, thresholds and temperature never see test data; the
  streaming output equals the offline causal pass, so offline evaluation of causal models is exactly
  what the device would output.

## 3. Detectors

| Detector | Input | Notes |
|---|---|---|
| Freeze Index (Moore et al., 2008) | ankle vertical axis | power 3-8 Hz / power 0.5-3 Hz over a causal 4 s window, plus a minimum-power gate; both thresholds chosen on validation patients |
| TCN-causal | 9 raw axes | 6 dilated causal residual layers, hidden 32 (receptive field 4 s), 12 epochs |
| TCN-spectral (selected) | 9 raw axes + 19 causal log band powers | 8 layers (16 s), 20 epochs |
| TCN-spectral + noise aug. | same | plus one noisy copy of each training recording (Gaussian noise, std U(0.1, 0.5) x feature std) |
| TCN-causal, deeper + noise aug. | 9 raw axes | 8 layers, 20 epochs, noise augmentation |
| TCN-acausal, LSTM, BiLSTM | 9 raw axes | comparison models; acausal ones re-run on the last 10 s every 250 ms online |

All learned models use class-weighted cross-entropy on 8 s windows (weight sqrt of the class ratio,
capped at 10) and AdamW with cosine decay. The causal TCN streams with per-layer caches: one new time
step per layer per frame, so the cost per frame is constant (`tests/test_pipeline.py`).

The three development variants target two weaknesses of the baseline: it had to rediscover the
frequency shift that defines freezing from 7 training patients, and it was fragile to sensor noise.
The variant used as "the" learned model is chosen by mean validation PR-AUC (validation patients
only); the others are reported as ablations. The spectral features are the Freeze Index's own ingredients, computed per
axis over 2 s, causally.

## 4. Metrics

* **Event metrics.** Predicted event = run of alarm frames, runs less than 1 s apart merged. Correct
  if at least 50% of its frames are freeze (precision = correct / predicted). An episode is detected if
  any alarm frame falls inside it (recall). False alarms per minute = incorrect events / minutes.
  Onset delay = time from episode onset to the first alarm frame inside it (detected episodes only).
* **Strict variant** (sensitivity check). One-to-one matching: a predicted event counts only if it is
  correct and overlaps an episode no earlier event matched; an episode counts as detected only if a
  correct event matched it. This removes two ways the default can be generous: splitting one freeze
  into several "correct" alarms, and detecting an episode with an alarm that is itself a false alarm.
* **PR-AUC** (average precision) per patient on frame scores. For the Freeze Index the score is the
  power-gated FI, the signal its detector thresholds.
* **Uncertainty.** Point estimates pool event counts over the 8 test patients within a seed and
  average the seeds. 95% CIs resample **patients** with replacement (2000 times) and recompute the
  pooled metric. Paired comparisons resample the same patients for both detectors. The exact Wilcoxon
  signed-rank test on per-patient values is reported next to it; with 8 patients its smallest
  two-sided p-value is 2/256 = 0.008.

## 5. Results

Online setting, threshold chosen on validation patients (`results/daphnet_summary.md`).

| Detector | Operating point | Event F1 [95% CI] | Strict event F1 | Recall | Precision | False alarms / min | PR-AUC [95% CI] | Debounce (median) |
|---|---|---|---|---|---|---|---|---|
| Freeze Index | threshold | 52.1 [40.7, 58.2] | 40.3 [30.5, 46.4] | 57.8 | 47.5 | 0.68 | 0.414 [0.290, 0.528] | 0.00 s |
| TCN-causal | threshold | 48.1 [34.0, 59.3] | 36.2 [23.5, 48.6] | 65.5 | 38.3 | 1.11 | 0.528 [0.429, 0.631] | 0.00 s |
| TCN-acausal | threshold | 47.5 [34.1, 57.0] | 35.9 [24.6, 46.5] | 61.0 | 39.9 | 1.00 | 0.520 [0.409, 0.638] | 0.00 s |
| LSTM | threshold | 26.7 [16.7, 39.5] | 18.5 [11.6, 28.4] | 57.0 | 17.6 | 1.63 | 0.376 [0.273, 0.479] | 0.00 s |
| BiLSTM | threshold | 35.7 [21.3, 49.0] | 24.6 [14.0, 36.8] | 61.6 | 25.2 | 1.60 | 0.363 [0.253, 0.486] | 0.00 s |
| TCN-causal, deeper + noise aug. | threshold | 47.8 [32.4, 61.0] | 35.1 [23.2, 46.6] | 63.0 | 38.7 | 0.85 | 0.496 [0.372, 0.617] | 0.00 s |
| TCN-spectral + noise aug. | threshold | 52.0 [35.9, 64.3] | 36.4 [22.7, 50.3] | 71.0 | 41.0 | 0.83 | 0.598 [0.501, 0.702] | 0.00 s |
| TCN-spectral | threshold | 58.5 [43.7, 68.1] | 40.8 [28.0, 53.3] | 70.7 | 50.1 | 0.63 | 0.641 [0.549, 0.735] | 0.00 s |

Paired with the Freeze Index:

| Detector | Operating point | Delta F1 [95% CI] | P(>0) | Wilcoxon p | Wins | Delta strict F1 [95% CI] | Delta PR-AUC [95% CI] | PR-AUC Wilcoxon p | PR-AUC wins |
|---|---|---|---|---|---|---|---|---|---|
| TCN-causal | threshold | -4.0 [-10.2, +3.1] | 0.18 | 0.945 | 4/8 | -4.1 [-11.4, +5.1] | +0.114 [+0.029, +0.193] | 0.055 | 6/8 |
| TCN-acausal | threshold | -4.6 [-9.3, +1.7] | 0.07 | 0.461 | 3/8 | -4.4 [-11.0, +4.4] | +0.106 [+0.029, +0.184] | 0.078 | 6/8 |
| LSTM | threshold | -25.4 [-30.6, -14.5] | 0.00 | 0.016 | 1/8 | -21.8 [-25.4, -14.2] | -0.038 [-0.139, +0.060] | 0.641 | 3/8 |
| BiLSTM | threshold | -16.4 [-24.3, -5.7] | 0.00 | 0.078 | 2/8 | -15.7 [-22.5, -6.0] | -0.051 [-0.127, +0.019] | 0.641 | 4/8 |
| TCN-causal, deeper + noise aug. | threshold | -4.3 [-14.1, +8.0] | 0.23 | 0.383 | 4/8 | -5.2 [-14.2, +7.0] | +0.082 [+0.012, +0.158] | 0.109 | 5/8 |
| TCN-spectral + noise aug. | threshold | -0.2 [-10.1, +9.3] | 0.53 | 0.742 | 4/8 | -3.8 [-11.3, +5.9] | +0.184 [+0.110, +0.262] | 0.008 | 8/8 |
| TCN-spectral | threshold | +6.3 [-3.0, +14.7] | 0.92 | 0.148 | 7/8 | +0.6 [-6.6, +9.2] | +0.227 [+0.141, +0.314] | 0.008 | 8/8 |

Reading the results:

1. **Ranking.** TCN-spectral separates freeze from non-freeze frames better than the Freeze Index in
   every one of the 8 patients (PR-AUC +0.227 [+0.141, +0.314], exact Wilcoxon p = 0.008). The baseline
   TCN already did so in 6 of 8 patients (+0.114 [+0.029, +0.193]).
2. **Alarms.** At one validation-chosen threshold, TCN-spectral has higher event F1 in 7 of 8 patients
   (58.5 vs 52.1; paired +6.3 [-3.0, +14.7]; bootstrap P(>0) = 0.92; Wilcoxon p = 0.148), detecting more
   episodes (70.7% vs 57.8%) with slightly fewer false alarms (0.63 vs 0.68 per minute). Likely, but not
   established with 8 patients. Under strict one-to-one matching the two are tied (40.8 vs 40.3).
3. **Timing.** The learned models flag onsets within tens to hundreds of milliseconds; the Freeze Index
   needs about a second because its 4 s window must fill with freeze-band power.
4. **Debounce.** Requiring an alarm to last 0.25 to 1.5 s (chosen on validation) lifts the Freeze Index to
   55.0 but does not help the learned models, whose probability is already smooth.

### Model development and selection

| Variant | Mean validation PR-AUC (selection) | Test PR-AUC | Test event F1 |
|---|---|---|---|
| TCN-causal (baseline) | 0.462 | 0.528 | 48.1 |
| TCN-causal, deeper + noise aug. | 0.486 | 0.496 | 47.8 |
| TCN-spectral + noise aug. | 0.589 | 0.598 | 52.0 |
| **TCN-spectral** | **0.615** | **0.641** | **58.5** |

Spectral input features are what helped: +0.10 validation PR-AUC at equal depth and augmentation
(0.486 to 0.589), +0.15 over the baseline overall. Noise augmentation lowered clean
accuracy and did not buy robustness to noise (section 6). The chosen variant is also the best on the
test patients; the two weakest variants swap places on test, within their CIs.

SEARCH_SECTION

### Offline vs. online
Causal models produce identical results offline and online. Acausal models re-run on the last 10 s
every 250 ms do not beat the causal TCN online (`results/daphnet/results.md`), and their per-update cost
grows with the history they re-process.

## 6. Calibration, abstention and robustness

From `results/daphnet/robustness.md` (baseline models; every choice on clean validation data):

* **Calibration.** Temperature scaling on two validation patients lowers test NLL (TCN-causal 0.421 to
  0.372) but not ECE (7.4% to 8.4%), and the fitted temperature varies across folds (2.6 ± 1.4).
  Probabilities are not calibrated risk for a new patient.
* **Abstention.** Withholding alarms when max(p, 1-p) < tau raises accuracy on covered frames (83% to
  92.5% at tau 0.7) but loses recall (66% to 40%); validation chose no abstention in all 24 TCN-causal runs.
* **Faults.** Up to 20% dropped frames and 3 s stuck sensors barely matter. Gaussian noise is the failure
  mode: at 1x feature std false alarms rise from 1.14 to 3.36 per minute for TCN-causal; the Freeze
  Index loses 22 F1 points, the TCN 13.

For the development variants (`results/daphnet_spectral/robustness.md`,
`results/daphnet_spectral_aug/robustness.md`): at noise 1x feature std, event F1 falls from 56.8 to 33.0
(TCN-spectral) and from 51.7 to 30.2 (with noise augmentation); the raw-input TCN keeps 35.5. Spectral
features import the Freeze Index's weakness, since noise adds freeze-band power. Augmentation with noise up
to 0.5x std reduced false alarms under heavy noise only slightly (2.87 vs 3.30 per minute). TCN-spectral's
temperature scaling changes ECE from 8.4% to 7.9% (temperature 2.0 ± 0.9 across folds).

## 7. Deployment

Each causal model exports as one streaming step (frame + state in, logits + state out) to TorchScript,
ONNX and dynamic int8 ONNX (`results/deployment.md`). For the baseline TCN (26.5k parameters), ONNX
Runtime fp32 takes 0.042 ms per frame on one CPU thread, about 10x faster than eager PyTorch; outputs
match the offline pass within 2e-6. Dynamic int8 halves the file size with a 0.2 point change in event
F1 but is slower at this model size. TCN-spectral (35k parameters) adds the spectral featurizer (one 64-point FFT per axis
per frame) and was not yet exported; its eager PyTorch step takes about 0.35 ms.

## 8. Session-report agent

`rtsurg/report_agent.py` is a LangGraph state machine: analyze (code) -> draft (Claude) -> verify (code)
-> revise (at most twice) or template fallback -> finalize. The verifier is the safety mechanism: every
number and time stamp in the draft must appear in the facts the code computed; diagnostic, treatment,
medication and dosing language is rejected; four fixed sections are required; quality flags must be
mentioned. Revision turns are appended to the conversation (never edited), and the full assistant
content is replayed. Refusals, API errors and repeated failures fall back to a deterministic template,
and every report carries a disclaimer. The graph is tested offline with scripted writers
(`tests/test_report_agent.py`); the Claude path needs an API key and was not exercised in CI.

## 9. Review notes (this revision)

Audit of the code and results before the changes above:

* **Corrected: Freeze Index PR-AUC.** It was computed on the raw FI score, while its detector alarms only
  when leg power exceeds a gate. Standing still gives a high FI ratio from noise, so the ungated score
  ranks those frames as freeze-like and its PR-AUC was understated (0.265). On the gated score it is
  0.414. The README's "about 2x the PR-AUC" was therefore overstated; the corrected baseline gap is 1.3x.
* **Checked, no leakage found.** Validation patients are disjoint from training and test; normalisation
  is fit on training patients; epoch, threshold, Freeze Index thresholds, temperature and abstention
  are chosen on validation patients; robustness reuses clean-validation choices. The JIGSAWS path trains
  for a fixed number of epochs and only logs the held-out loss.
* **Event metric.** The definition is reasonable and stops an always-on detector, but it can be generous
  (fragmented alarms, detection by an alarm that is itself a false alarm). The strict variant now
  reported leaves the ranking of detectors unchanged; strict F1 is 8 to 16 points lower for every
  detector. Learned models lose more than the Freeze Index (TCN-spectral 58.5 to 40.8, Freeze Index 52.1
  to 40.3) because they sometimes split a long freeze into several alarms; under strict matching
  TCN-spectral and the Freeze Index are tied (+0.6, CI -6.6 to +9.2).
* **Robustness to a NaN validation score.** `train_fog_model` crashed if every validation PR-AUC was NaN;
  it now keeps the last epoch.
* **Reproducibility.** The baseline re-run reproduced all 216 result rows (event counts and thresholds)
  of the original run exactly. Its latency block was kept from the original run because the re-run
  shared the CPU with other jobs.
* **README claims.** Every other number in the previous README matched `results/`.

## 10. Limitations and next steps

* 8 test patients, one lab protocol, scripted walking tasks. A second dataset (for example the
  Michael J. Fox Foundation tDCS FoG and DeFOG recordings released on Kaggle, lower-back sensor at
  100-128 Hz) is the most valuable next step: train on one, test on the other.
* Faults are simulated; no device or patient testing.
* Probabilities are not calibrated for new patients; per-patient calibration from a short enrolment
  recording is an open question.
* JIGSAWS results await data access; the LOUO and LOSO pipelines are tested on generated files only.
