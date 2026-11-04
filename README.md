JIGSAWS results pending dataset access; real results currently on Daphnet Freezing of Gait.

[![tests](https://github.com/ZeynepHusnaKocacenk/realtime-surgical-gesture-jigsaws/actions/workflows/tests.yml/badge.svg)](https://github.com/ZeynepHusnaKocacenk/realtime-surgical-gesture-jigsaws/actions/workflows/tests.yml)

# From Offline to the Operating Room: Real-Time Surgical Gesture and Skill Recognition on JIGSAWS

Most surgical gesture recognition results on JIGSAWS are reported **offline**: the model sees the
entire recording, then labels every frame. A system that coaches or monitors a surgeon during a
procedure cannot do that. At time *t* it only has frames 0..*t*, and it has a fixed time budget to
answer before the next frame arrives.

This project measures that gap directly. It trains causal and acausal sequence models on JIGSAWS
robot kinematics under the standard Leave-One-User-Out protocol and evaluates each one twice:
offline, as in the literature, and **online**, under the constraints of a live system.

It extends my undergraduate thesis at Carleton University, a real-time surgical skill classification
pipeline (PyTorch BiLSTM/TCN inside 3D Slicer, under 25 ms inference) built on simulated motion
tracking data ([repo](https://github.com/ZeynepHusnaKocacenk/realtime-surgical-skill-analysis-3dslicer)).
Here the same model families are tested on real da Vinci kinematics with the benchmark's standard
evaluation protocol.

## Questions this project answers

1. **How much accuracy do the usual offline models lose when they cannot see the future?**
2. **Which architectures fit a real-time budget**, and does their latency stay constant as a
   procedure gets longer?
3. **How quickly after a gesture starts** does an online model recognise it?
4. **How early in a trial** can a surgeon's skill level be estimated?

## Method

### Data and protocol
* **JIGSAWS** (Gao et al., 2014): da Vinci kinematics at 30 Hz from 8 surgeons of varying skill
  performing Suturing, Knot-Tying and Needle-Passing, with frame-level gesture labels and
  self-reported skill (Novice / Intermediate / Expert).
* **Features:** the 38 patient-side manipulator (PSM) variables by default (position, rotation,
  linear and angular velocity, gripper angle for both arms). Master-side variables can be added
  with `--manipulators`.
* **Leave-One-User-Out (LOUO):** each fold holds out every trial of one surgeon. No surgeon appears
  in both training and test data, so models cannot succeed by recognising an individual's style.
  Normalisation statistics are fitted on the training fold only. Models train for a fixed number of
  epochs, and the held-out surgeon is never used for early stopping or model selection.
* Kinematics are downsampled to 10 Hz by default, which gives a **100 ms per-frame budget**.

### Models
All models share the same input projection, a per-frame gesture head and a skill head on a running
mean of frame features, so they differ only in their temporal backbone.

| Model | Sees the future? | How it runs live |
|---|---|---|
| **TCN-causal** | No | Dilated causal convolutions with per-layer caches; one new time step per layer per frame |
| **LSTM** (uni-directional) | No | Carries its hidden state; constant cost per frame |
| **TCN-acausal** | Yes (centred kernels) | Must be re-run on the full history |
| **BiLSTM** | Yes (backward pass) | Must be re-run on the full history |

The TCN follows the single-stage MS-TCN design (Farha & Gall, 2019): dilated residual layers with a
temporal smoothing loss that reduces over-segmentation. Gesture recognition and skill assessment are
learned jointly.

### Efficient streaming for the causal TCN
`TCN.stream_step` does not re-run the network on a window of past frames. Each dilated layer keeps a
cache of its last `(kernel-1)*dilation` inputs (zeros at the start, which is exactly the causal zero
padding of the offline pass). A new frame computes **one** new time step per layer from that cache,
so the work per frame is a fixed number of multiply-accumulates: it grows linearly with depth, not
with the receptive field (which doubles with every layer), and it does not depend on how long the
procedure has run. `tests/test_pipeline.py` checks that the streamed outputs equal the offline pass
and that the per-frame cost is identical at every frame.

### Offline vs. online evaluation
* **Offline:** each model labels a trial after seeing all of it.
* **Online:** causal models produce exactly the same outputs live as offline. This is verified by a
  unit test that streams frames one at a time through `stream_step` and checks the result against the
  offline pass. Acausal models are re-run on the history every *k* frames (`--recompute-stride`) and
  hold their last answer in between, as a deployed system would have to.

### Metrics
| Metric | What it measures |
|---|---|
| Frame accuracy | % of frames labelled correctly |
| Edit score | Segment-order correctness; penalises flickering predictions (Lea et al., 2016) |
| F1@{10, 25, 50} | Segment detection at IoU thresholds (Lea et al., 2017) |
| Per-update latency | Median and p95 ms at batch size 1, measured at 10 s to 5 min of elapsed procedure time |
| Detection delay | Time from a gesture's onset until the online model first outputs the correct label |
| Early skill accuracy | Skill accuracy after observing 10%, 25%, 50%, 75%, 100% of a trial |

## Results

> **To be filled in after running on JIGSAWS.** `python main.py --data-dir <path>/Suturing` writes
> `results/results.md` (tables) and five figures to `results/figures/`.

Figures produced:
* `accuracy_vs_latency.png`: online accuracy against per-update latency, with the frame budget marked
* `offline_vs_online.png`: what each model loses when restricted to past frames
* `latency_vs_history.png`: latency as the procedure gets longer (causal flat, acausal growing)
* `gesture_timeline.png`: ground truth vs. online predictions on a held-out surgeon
* `early_skill.png`: skill accuracy vs. fraction of the trial observed

## Results: Daphnet Freezing of Gait

Real data: the Daphnet dataset (Bachlin et al., 2010; UCI), 10 Parkinson's patients, three 3-axis
accelerometers (ankle, upper leg, trunk), 64 Hz. These are the results this project currently has;
JIGSAWS results are pending dataset access. Full tables are in
[`results/daphnet/results.md`](results/daphnet/results.md), raw per-fold numbers in
`results/daphnet/results.json`.

### Protocol
* **Data:** frames annotated "not part of experiment" are dropped, and recordings are split wherever
  they were interrupted, so no sequence spans a gap. Signals are averaged in pairs to 32 Hz (a causal
  operation), giving a **31.3 ms** frame budget. Normalisation uses the training subjects only.
* **Leave-one-subject-out, with validation subjects.** Each fold holds out one subject. Two further
  subjects (that have freezes, drawn by seed) are held out of the training set as **validation
  subjects**. They are used for epoch selection (best validation PR-AUC) and for **every decision
  threshold**; the test subject is never used for any choice. Test subjects with no freeze episodes
  (S04, S10) are skipped, so 8 test folds remain; S04 and S10 are still used for training.
* **Seeds:** 3 (0, 1, 2). The validation subjects, initialisation and data order change with the seed.
  Event metrics are pooled over the 8 test subjects within a seed and reported as mean ± std across
  seeds. Sensitivity, specificity and PR-AUC are per fold, mean ± std across seeds and folds.
* **Models:** the four models from above (hidden size 32, 6 TCN layers, 12 epochs, class-weighted
  cross-entropy on 8 s windows). Online acausal models are re-run on the last 10 s every 250 ms.
* **Non-learned baseline:** the Freeze Index (Moore et al., 2008): power in 3 to 8 Hz divided by power
  in 0.5 to 3 Hz on the ankle vertical axis, over a causal 4 s window, thresholded. The FI threshold
  and a minimum-power threshold are chosen on the validation subjects.
* **Metrics.** Freeze frames are the minority class (**12.8%** of test frames; 237 episodes), so frame
  accuracy is not the headline. A *predicted event* is a run of positive frames, merging runs less than
  1 s apart. An episode is *detected* if any positive frame falls inside it (recall); a predicted event
  is *correct* only if at least half of its frames are freeze (precision), so an always-on detector
  scores zero precision. Also reported: sensitivity, specificity, PR-AUC, false alarms per minute, and
  detection delay (onset to first positive frame).

### Online results (deployable setting; mean ± std across 3 seeds)

| Model | Causal | Event precision | Event recall | Event F1 | PR-AUC | Median onset delay (ms) | False alarms / min | Latency median (ms) |
|---|---|---|---|---|---|---|---|---|
| Freeze Index (baseline) | yes | 47.5 ± 2.6 | 57.8 ± 2.8 | 52.1 ± 2.6 | 0.265 ± 0.122 | 1094 | 0.68 ± 0.04 | n/a |
| TCN-causal | yes | 38.3 ± 3.4 | 65.5 ± 8.4 | 48.1 ± 3.9 | 0.528 ± 0.167 | 250 | 1.11 ± 0.07 | 0.49 |
| TCN-acausal | no | 39.9 ± 8.3 | 61.0 ± 6.2 | 47.5 ± 5.9 | 0.520 ± 0.177 | 359 | 1.00 ± 0.18 | 3.08 |
| LSTM | yes | 17.6 ± 2.3 | 57.0 ± 2.1 | 26.7 ± 2.6 | 0.376 ± 0.186 | 0 | 1.63 ± 0.16 | 0.16 |
| BiLSTM | no | 25.2 ± 2.4 | 61.6 ± 6.8 | 35.7 ± 3.4 | 0.363 ± 0.195 | 156 | 1.60 ± 0.19 | 17.4 |

Offline vs. online event F1: TCN-causal 48.1 vs 48.1 (identical by construction), TCN-acausal 48.0 vs
47.5, LSTM 26.7 vs 26.7, BiLSTM 31.8 vs 35.7. Latency is the median per update at 60 s of elapsed time
for acausal models re-run on the whole history (the causal models do not depend on elapsed time); the
online acausal evaluation above re-runs them on a bounded 10 s window instead (TCN-acausal 1.5 ms,
BiLSTM 3.0 ms per update at 10 s).

### What the results say
* The **causal TCN has the best PR-AUC** (0.53, about twice the Freeze Index's 0.27) and detects
  freeze onsets about **0.8 s earlier** than the Freeze Index (median 250 ms vs 1094 ms), at 0.5 ms
  per frame, constant as the recording gets longer.
* At the validation-selected operating points, its **event F1 (48.1 ± 3.9) is not better than the
  Freeze Index (52.1 ± 2.6)**: the learned model finds more episodes (recall 65.5% vs 57.8%) at the cost
  of more false alarms (1.1 vs 0.7 per minute). With 3 seeds the difference is within noise.
* The causal TCN loses nothing when restricted to the past; the LSTM variants are clearly worse on
  this task.
* Results vary a lot between subjects (per-subject event F1 from 14 to 70 for the causal TCN), as in
  the literature; see the per-subject tables in `results/daphnet/results.md`.

Figures: `results/daphnet/figures/offline_vs_online.png`, `per_subject_f1.png`,
`latency_vs_history.png`, `recall_vs_false_alarms.png`.

### Limitations
* Small cohort (10 patients, 8 evaluable), recorded in a lab with scripted walking tasks designed to
  provoke freezes. Results do not transfer to free-living use and are **not clinical claims**.
* Models are small and trained for a fixed 12 epochs with no hyper-parameter search, so the learned
  models may be under-tuned; the event metrics depend on the merge-gap and the 50% rule above.
* Three seeds give only a rough estimate of variability, and the validation subjects change with the
  seed. Latency was measured on a shared CPU container.
* Detection delay is measured only for detected episodes.

## Robustness and safe abstention

Real Daphnet data, the same 96 trained models as above (3 seeds x 8 folds x 4 models), no retraining:
`python main.py --dataset daphnet --robustness --data-dir <daphnet folder>`. Full tables:
[`results/daphnet/robustness.md`](results/daphnet/robustness.md). Figures:
`results/daphnet/figures/robustness.png` and `abstention.png`. Every choice (temperature, decision
threshold, abstention threshold, Freeze Index thresholds) was made on the validation subjects of the
fold, on clean data. Thresholds are **not** re-tuned for faulty data.

**Calibration.** Temperature scaling is fitted on validation subjects (`rtsurg/calibration.py`).
It lowers the negative log-likelihood on the test subjects (TCN-causal 0.421 to 0.372, LSTM 0.526 to
0.446) but **does not reduce ECE**: TCN-causal 7.4% before, 8.4% after (std about 4.5 across folds);
LSTM 12.9% to 12.5%. The fitted temperature varies a lot between folds (TCN-causal 2.6 ± 1.4), so
what is learned on two validation subjects does not transfer reliably to a new patient. The models
are over-confident on unseen subjects, and this calibration step only partly fixes it.

**Sensor faults** (online recognition, test subjects only; drops hold the last frame, noise is Gaussian
with std relative to each feature's training std, a stuck sensor repeats its value for 1 to 3 s,
one event per 30 s of recording at random times). Freeze event F1, mean ± std over 3 seeds (the clean
row uses the temperature-scaled threshold, so it differs slightly from the main table above):

| Condition | Freeze Index | TCN-causal | TCN-acausal | LSTM | BiLSTM |
|---|---|---|---|---|---|
| clean | 52.1 ± 2.6 | 48.8 ± 4.2 | 47.8 ± 6.6 | 25.5 ± 2.6 | 36.3 ± 2.5 |
| 20% dropped frames | 49.9 ± 2.5 | 48.7 ± 2.9 | 45.0 ± 5.4 | 27.5 ± 2.1 | 36.7 ± 0.8 |
| noise 0.5x std | 36.2 ± 2.9 | 42.2 ± 1.9 | 40.5 ± 3.7 | 26.2 ± 1.2 | 32.5 ± 1.7 |
| noise 1.0x std | 30.0 ± 1.8 | 35.5 ± 3.3 | 34.9 ± 0.6 | 27.0 ± 1.6 | 31.1 ± 3.7 |
| sensor stuck 3 s | 51.0 ± 2.2 | 47.0 ± 3.8 | 46.2 ± 7.3 | 25.1 ± 2.6 | 35.9 ± 2.4 |

* Dropped frames (up to 20%) and short stuck-sensor episodes (up to 3 s) change the results very little,
  because the models integrate over several seconds.
* **Noise is the real failure mode.** At 1.0x std, false alarms rise from 1.1 to 3.4 per minute for
  TCN-causal (clean 256 false alarms, 758 at 1.0x std over the 8 test subjects). Recall appears to
  rise because noise triggers alarms, so read it together with false alarms. The Freeze Index loses
  22 F1 points at 1.0x noise, the causal TCN 13. The LSTM models are low to begin with and move little.
  Missed freezes and false alarms for every condition are in the last table of `robustness.md`.

**Abstention.** After temperature scaling, a frame whose top-class probability is below tau is
"uncertain" and raises no alarm (tau from 0.5 to 0.9). For TCN-causal, tau 0.7 keeps 78% of frames,
raises accuracy on those frames from 83% to 92.5%, and halves false alarms (1.14 to 0.49 per minute),
but event recall falls from 66% to 40% (more missed freezes: 81 to 143 of 237). At tau 0.9 only 55%
of frames get a decision and recall is 12%. Choosing tau on validation subjects (best event F1 with
coverage of at least 80%) **selected tau 0.5 (no abstention) in all 24 TCN-causal runs and in most others**, so abstention is a way to
trade missed freezes for fewer false alarms, not a free improvement. Whether that trade is acceptable
is a clinical question this project does not answer.

Limitations: faults are simulated, not recorded from failing sensors; the injection pattern (one stuck
event per 30 s, independent frame drops) is one of many possible; and three seeds give rough error bars.

## Deployment

A design proposal, with **no clinical claims**: how a trained causal model could run on a wearable
gateway. `python export.py --data-dir <daphnet folder>` exports each causal model as a single
streaming step (new frame plus model state in, freeze logits plus new state out) to TorchScript and
ONNX, quantises the ONNX model to dynamic int8, checks the outputs against the offline PyTorch pass
(max absolute difference at most 2e-6 for fp32), and benchmarks CPU latency with one thread. Results:
[`results/deployment.md`](results/deployment.md), figure `results/deployment_latency.png`.

| TCN-causal (26.5k parameters) | Median ms / frame | Size (MB) | Event F1 (8 test subjects, seed 0) |
|---|---|---|---|
| PyTorch eager | 0.412 | 0.118 | not run |
| TorchScript | 0.096 | 0.119 | not run |
| ONNX Runtime fp32 | 0.042 | 0.115 | 52.6 |
| ONNX Runtime dynamic int8 | 0.084 | 0.064 | 52.4 (-0.2) |

Every runtime is far inside the 31.3 ms frame budget (ONNX Runtime is about 10x faster than eager
PyTorch). int8 halves the model size at no measurable accuracy cost here (-0.2 F1 for TCN-causal, +0.2
for the LSTM), but at this tiny model size it is **slower** than fp32 because of quantisation
overhead, so its only benefit is memory. Proposed design: a ring buffer of the three sensors at 64 Hz,
2:1 averaging to 32 Hz, one ONNX step per frame with state kept in memory, a decision threshold chosen
on held-out patients, and the abstention and fault results above as the starting points for a safety
analysis. None of this has been tested on a device or with patients.

## Running it

Streaming demo (**SYNTHETIC DATA, pipeline demo, not a result**): `python demo_stream.py` trains a
small causal TCN on generated data, streams one held-out trial frame by frame, and prints the
predicted and true gesture, per-frame latency, and how many frames stayed within the 100 ms budget.

```bash
pip install -r requirements.txt
python -m pytest -q tests                     # correctness tests, about 1 min

# Pipeline check without the dataset: generates mock data in the exact JIGSAWS file layout.
# SYNTHETIC DATA, pipeline demo, not a result: these numbers are meaningless, they only show
# that everything runs.
python main.py --synthetic --epochs 15 --max-folds 3

# Daphnet (folder holding the SxxRyy.txt files; the data is not included in this repository)
python main.py --dataset daphnet --data-dir /path/to/dataset_fog_release/dataset

# Real JIGSAWS (one task folder, all 8 LOUO folds)
python main.py --data-dir /path/to/JIGSAWS/Suturing --epochs 60 --verbose
```

Useful options: `--models TCN-causal LSTM`, `--downsample 1` (full 30 Hz),
`--recompute-stride 1` (re-run acausal models every frame, slower to evaluate),
`--latency-device cuda`, `--manipulators MTM_L MTM_R PSM_L PSM_R`.

JIGSAWS is distributed by Johns Hopkins University for non-commercial research and is not included
in this repository.

## Project structure

```
rtsurg/
  data.py        JIGSAWS loader, synthetic generator (same file format), LOUO splits, normalisation
  models.py      causal/acausal TCN, LSTM/BiLSTM, frame-by-frame streaming API
  train.py       joint gesture + skill training with temporal smoothing loss
  metrics.py     frame accuracy, edit score, F1@k, detection delay
  benchmark.py   offline vs. online evaluation, early skill curve, latency profiler
  daphnet.py     Daphnet loader, leave-one-subject-out splits with validation subjects, Freeze Index
  fog.py         event-level metrics, threshold selection, windowed training, online evaluation
  fog_run.py     Daphnet benchmark driver (3 seeds in parallel)
  fog_report.py  Daphnet tables and figures
  calibration.py temperature scaling and ECE
  robustness.py  sensor-fault injectors
  robust_run.py / robust_report.py  calibration, fault and abstention evaluation
export.py        ONNX / TorchScript / int8 export and latency benchmark
  plots.py       figures
main.py          runs the full LOUO benchmark and writes tables and figures
tests/           loader, leakage, metric, causality and streaming-equivalence tests
```

## Limitations

* JIGSAWS is small (8 surgeons, 39 Suturing trials), so fold-to-fold variance is large; results are
  reported as mean ± standard deviation over folds.
* Under LOUO, skill classification depends on very few surgeons per skill level (e.g. 2 experts), which
  makes it much harder than the commonly reported leave-one-supertrial-out setting. Skill results here
  should be read as a stress test, not compared directly with LOSO numbers.
* Latency is measured on the machine that runs the benchmark; the hardware is recorded in
  `results/results.json`.
* Kinematics only. Video features are a natural next step.

## References

* Y. Gao et al. *JHU-ISI Gesture and Skill Assessment Working Set (JIGSAWS): A Surgical Activity
  Dataset for Human Motion Modeling.* MICCAI Workshop M2CAI, 2014.
* C. Lea, A. Reiter, R. Vidal, G. Hager. *Segmental Spatiotemporal CNNs for Fine-grained Action
  Segmentation.* ECCV, 2016.
* C. Lea et al. *Temporal Convolutional Networks for Action Segmentation and Detection.* CVPR, 2017.
* Y. Abu Farha, J. Gall. *MS-TCN: Multi-Stage Temporal Convolutional Network for Action
  Segmentation.* CVPR, 2019.

## Author

Zeynep Kocacenk, B.Sc. Computer Science (Psychology minor), Carleton University.
