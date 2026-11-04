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

## Running it

Streaming demo (**SYNTHETIC DATA, pipeline demo, not a result**): `python demo_stream.py` trains a
small causal TCN on generated data, streams one held-out trial frame by frame, and prints the
predicted and true gesture, per-frame latency, and how many frames stayed within the 100 ms budget.

```bash
pip install -r requirements.txt
python -m pytest -q tests                     # 11 correctness tests, about 10 s

# Pipeline check without the dataset: generates mock data in the exact JIGSAWS file layout.
# SYNTHETIC DATA, pipeline demo, not a result: these numbers are meaningless, they only show
# that everything runs.
python main.py --synthetic --epochs 15 --max-folds 3

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
