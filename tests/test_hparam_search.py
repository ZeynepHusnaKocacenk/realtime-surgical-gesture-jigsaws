"""The hyperparameter search must never featurize, predict or score a fold's test patient."""
import numpy as np

import rtsurg.hparam_search as hs


def _write(path, ann, seed=0):
    rng = np.random.default_rng(seed)
    raw = np.column_stack([np.arange(len(ann)) * 15.6, rng.normal(size=(len(ann), 9)), ann])
    np.savetxt(path, raw, fmt="%.4f")


def test_search_never_touches_the_test_patient(tmp_path, monkeypatch):
    for i, s in enumerate(("S01", "S02", "S03", "S04")):
        ann = np.r_[np.full(400, 1), np.full(150, 2), np.full(300, 1)] if s != "S04" else np.full(800, 1)
        _write(tmp_path / f"{s}R01.txt", ann, seed=i)
    seen, tests = [], []
    real_featurize, real_splits = hs.featurize, hs.loso_splits

    def spy_featurize(x, kind, rate):
        seen.append((len(tests) - 1, x))          # tagged with the fold currently being run
        return real_featurize(x, kind, rate)

    def spy_splits(*a, **k):
        for user, train, val, test in real_splits(*a, **k):
            tests.append(test)
            yield user, train, val, test

    monkeypatch.setattr(hs, "featurize", spy_featurize)
    monkeypatch.setattr(hs, "loso_splits", spy_splits)
    cfg = dict(hs.BASE, hidden=8, layers=2, epochs=1, window=64)
    out = hs.run_config(cfg, str(tmp_path))
    assert len(out["folds"]) == 3 and all(f["val_pr_auc"] == f["val_pr_auc"] for f in out["folds"])
    for fold_tests, fold in zip(tests, out["folds"]):
        assert fold["fold"] not in fold["val_users"]
    # within each fold, none of that fold's test recordings is featurized (in other folds the same
    # patient is a legitimate training patient)
    for k, ts in enumerate(tests):
        assert not any(x is t.kinematics for f, x in seen if f == k for t in ts)
    assert {f for f, _ in seen} == set(range(len(tests)))


def test_grid_has_unique_names_and_contains_the_centre():
    names = [hs.name(c) for c in hs.grid()]
    assert len(names) == len(set(names)) and hs.name(hs.BASE) in names
