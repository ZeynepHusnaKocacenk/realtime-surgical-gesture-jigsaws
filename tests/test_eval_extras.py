"""Strict event matching, debouncing, causal spectral features and the statistics helpers."""
import numpy as np

from rtsurg.features import spectral_features
from rtsurg.fog import event_counts, fi_gated_score, postprocess, select_operating_point, summarize_counts
from rtsurg.stats import bootstrap_metric, paired_comparison, wilcoxon_exact

RATE = 10.0


def _truth():
    t = np.zeros(400, int); t[100:200] = 1; t[300:340] = 1
    return t


def test_fragmented_alarms_are_rewarded_by_default_but_not_by_strict_matching():
    t = _truth()
    p = np.zeros(400, bool)
    for s in range(100, 200, 25):          # one freeze split into 4 alarms, 1.5 s apart (> merge gap)
        p[s:s + 10] = True
    loose = summarize_counts(event_counts([p], [t], RATE), RATE)
    strict = summarize_counts(event_counts([p], [t], RATE, strict=True), RATE)
    assert loose["event_precision"] == 100.0             # 4 "correct" events
    assert strict["event_precision"] == 25.0             # only one may match the episode
    assert loose["event_recall"] == strict["event_recall"] == 50.0


def test_strict_recall_needs_a_correct_event():
    t = _truth()
    p = np.zeros(400, bool); p[60:120] = True              # mostly before the freeze: a false alarm
    assert event_counts([p], [t], RATE)["detected"] == 1
    assert event_counts([p], [t], RATE, strict=True)["detected"] == 0


def test_postprocess_is_causal_and_drops_short_runs():
    p = np.zeros(50, bool); p[5:7] = True; p[20:30] = True
    q = postprocess(p, 4)
    assert not q[5:7].any()                                 # 2-frame blip removed
    assert not q[20:23].any() and q[23:30].all()            # alarm starts once 4 frames have been seen
    for t in range(50):                                     # output at t depends on p[:t+1] only
        p2 = p.copy(); p2[t + 1:] = ~p2[t + 1:]
        assert postprocess(p2, 4)[t] == q[t]
    assert (postprocess(p, 0) == p).all()


def test_operating_point_selection_prefers_debounce_on_flickery_scores():
    rng = np.random.default_rng(0)
    t = _truth()
    prob = 0.8 * t + 0.1 + 0.0 * rng.random(400)
    prob[rng.choice(np.flatnonzero(t == 0), 20, replace=False)] = 0.95     # isolated spikes
    thr, md = select_operating_point([prob], [t], RATE, min_durs_s=(0.0, 0.5))
    assert md == 5


def test_fi_gate_zeroes_standing_frames():
    assert (fi_gated_score(np.array([5.0, 5.0]), np.array([0.0, 2.0]), 1.0) == [0.0, 5.0]).all()


def test_spectral_features_are_causal_and_shaped():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 9))
    f = spectral_features(x, 32.0)
    assert f.shape == (300, 19) and np.isfinite(f).all()
    x2 = x.copy(); x2[200:] = rng.normal(size=(100, 9)) * 5
    f2 = spectral_features(x2, 32.0)
    assert np.allclose(f[:200], f2[:200])                   # future samples never matter
    assert not np.allclose(f[200:], f2[200:])


def test_freeze_band_feature_responds_to_trembling():
    t = np.arange(640) / 32.0
    walk = np.sin(2 * np.pi * 1.5 * t); tremble = np.sin(2 * np.pi * 6.0 * t)
    x = np.zeros((640, 9)); x[:320, 1] = walk[:320]; x[320:, 1] = tremble[320:]
    f = spectral_features(x, 32.0)
    assert f[600, 18] > f[200, 18] + 3                        # log Freeze Index of the ankle rises


def test_wilcoxon_exact_matches_known_values():
    assert abs(wilcoxon_exact([1, 2, 3, 4, 5, 6, 7, 8]) - 2 / 256) < 1e-12
    assert wilcoxon_exact([1, -1]) == 1.0
    assert wilcoxon_exact([0, 0]) == 1.0


def _fake_rows():
    t = _truth()
    rows = []
    for seed in (0, 1):
        for i, f in enumerate(["A", "B", "C", "D"]):
            good = np.zeros(400, bool); good[110:200] = True; good[305:340] = True
            bad = np.zeros(400, bool); bad[110:200] = True; bad[0:20] = True; bad[230:250] = True
            for m, p in (("good", good), ("bad", bad)):
                rows.append({"seed": seed, "fold": f, "model": m, "mode": "online",
                             "counts": event_counts([p], [t], RATE)})
    return rows


def test_bootstrap_and_paired_comparison():
    rows = _fake_rows()
    g = bootstrap_metric(rows, "good", "online", RATE, n_boot=200)
    assert g["point"] == 100.0 and g["lo"] == g["hi"] == 100.0       # identical subjects -> no spread
    d = paired_comparison(rows, "good", "bad", RATE, n_boot=200)
    assert d["diff"] > 0 and d["lo"] > 0 and d["p_boot_gt0"] == 1.0 and d["wins"] == 4
    assert abs(d["wilcoxon_p"] - 2 / 16) < 1e-12                       # 4 subjects, all positive
