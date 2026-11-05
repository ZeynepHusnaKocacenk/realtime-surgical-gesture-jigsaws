# Model card: causal TCN freezing-of-gait detector (research prototype)

Format follows Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019). This model is a
**research prototype**. It is not a medical device, has not been evaluated prospectively, and
nothing below is a clinical claim.

## Model details
* **Architecture.** Causal temporal convolutional network (single-stage MS-TCN style, dilated
  residual layers, kernel 3). The card describes **TCN-spectral, tuned**, selected by validation PR-AUC
  among five variants and a 16-configuration hyperparameter search: 64 channels, 10 layers (64 s receptive
  field), raw accelerations plus 19 causal log band powers from `rtsurg/features.py`, 20 epochs, 171k
  parameters. The baseline `TCN-causal`
  (6 layers, raw accelerations, 12 epochs, 26.5k parameters) is reported for comparison.
* **Output.** p(freeze) for every 31.25 ms frame. An alarm is raised when p(freeze) stays above a
  threshold for a minimum duration; both are chosen on validation patients.
* **Streaming.** Per-layer caches, one new time step per layer per frame; streamed outputs equal the
  offline pass (tested), and the cost per frame does not grow with recording length.
* **Size and speed.** About 1.0 ms per frame in eager PyTorch on a CPU, against a 31.25 ms frame budget.
  The baseline TCN runs in 0.04 ms with ONNX Runtime fp32 (`results/deployment.md`); the selected model
  has not been exported yet.
* **Code.** `rtsurg/models.py`, `rtsurg/fog.py`, `rtsurg/fog_run.py`; configs in `configs/`.

## Intended use
* Research on real-time detection of freezing-of-gait episodes from body-worn accelerometers, and on
  how to evaluate such detectors honestly (patient-level splits, event metrics, uncertainty).
* **Out of scope:** any clinical decision, cueing patients, unsupervised home use, people other than
  the population described below, sensor placements or sampling rates other than Daphnet's.

## Training and evaluation data
* **Daphnet Freezing of Gait** (Bachlin et al., 2010; UCI, CC BY 4.0): 10 people with Parkinson's
  disease, three 3-axis accelerometers (ankle, thigh, trunk) at 64 Hz (used at 32 Hz), scripted
  walking tasks in a lab designed to provoke freezing, annotated from video. 8 patients have
  freezing episodes and serve as test patients (237 episodes, 225 minutes).
* **Protocol.** Leave one patient out; in every fold two other patients with freezes are held out
  of training as validation patients and are the only data used to choose the epoch, threshold,
  debounce, temperature and abstention level. The test patient is never used for any choice.

## Performance
See `results/daphnet_summary.md` for every detector with 95% bootstrap CIs over test patients,
and `results/daphnet/results.md` for per-patient results. Event F1 counts a predicted alarm as
correct only if at least half of its frames are annotated freeze.

Online, 8 held-out patients x 3 seeds, threshold chosen on validation patients; [95% CI] over patients.

| Metric | Freeze Index (reference) | TCN-causal (baseline) | TCN-spectral, tuned (selected) |
|---|---|---|---|
| Event F1 | 52.1 [40.7, 58.2] | 48.1 [34.0, 59.3] | 61.2 [48.5, 68.4] |
| Event F1, strict one-to-one matching | 40.3 [30.5, 46.4] | 36.2 [23.5, 48.6] | 42.0 [30.5, 51.5] |
| Episode recall (%) | 57.8 | 65.5 | 72.3 |
| Event precision (%) | 47.5 | 38.3 | 53.1 |
| False alarms per minute | 0.68 | 1.11 | 0.61 |
| PR-AUC (mean over patients) | 0.414 [0.290, 0.528] | 0.528 [0.429, 0.631] | 0.599 [0.489, 0.705] |
| Median onset delay | 1094 ms | 250 ms | 203 ms |
| Per-patient event F1, range | 20 to 64 | 14 to 70 | 23 to 76 |

Paired with the Freeze Index over the same patients, the selected model has higher PR-AUC in 8 of 8
patients (+0.185 [+0.108, +0.270], exact Wilcoxon p = 0.008) and higher event F1 in 7 of 8 (+9.1
[+2.8, +14.5], Wilcoxon p = 0.148 because patient S01 goes the other way): likely better at the alarm
level, not proven with 8 patients.

## Known failure modes and limitations
* **Between-patient variation dominates.** Per-patient event F1 ranges from 23 to 76 (patient S01 is
  hard for every learned model); a CI over 8 patients is wide, and 8 patients cannot establish superiority.
* **Sensor noise** is the main failure mode: with Gaussian noise of 1x the feature std, false alarms rise
  from 0.64 to 3.25 per minute and event F1 falls from 60.4 to 33.7 (`results/daphnet_tuned/robustness.md`;
  Freeze Index 52.1 to 30.0); spectral features share part of the Freeze Index's noise weakness. Dropped frames and short stuck-sensor periods
  matter little. Faults were simulated, not recorded.
* **Calibration.** Temperature scaling on two validation patients lowers ECE only slightly (8.8% to
  7.4%) and the fitted temperature varies across folds (2.1 ± 1.0); probabilities should not be read as calibrated risk for a new patient.
* **Population and setting.** Lab recordings of one small cohort with scripted tasks; no evidence
  about free-living use, other devices, or other patient groups. No subgroup analysis (age, sex,
  disease stage) is possible with this data.
* **Labels.** Episode boundaries come from video annotation and are uncertain by a few hundred ms;
  event metrics depend on the 1 s merge gap and the 50% rule (a strict one-to-one matching is
  reported as a sensitivity check).

## Ethical considerations
* A missed freeze and a false alarm have different, context-dependent costs (fall risk vs. alarm
  fatigue); the operating point here maximises event F1 on validation data, which is a research
  convenience, not a clinical choice.
* The session-summary agent (`rtsurg/report_agent.py`) only rephrases numbers computed by code; a
  deterministic verifier rejects any number not in the detector output and any diagnostic,
  treatment or medication language, and every summary carries a disclaimer.
