"""Leave-one-supertrial-out split for JIGSAWS (pipeline test on generated files in the JIGSAWS layout)."""
from rtsurg.data import generate_synthetic_jigsaws, load_jigsaws_task, loso_supertrial_splits, supertrial


def test_loso_holds_out_one_repetition_of_every_surgeon(tmp_path):
    data = load_jigsaws_task(generate_synthetic_jigsaws(tmp_path, trials_per_user=3, seed=0))
    folds = list(loso_supertrial_splits(data))
    assert [f for f, _, _ in folds] == ["ST1", "ST2", "ST3"]
    seen = set()
    for name, train, test in folds:
        r = int(name[2:])
        assert {supertrial(t) for t in test} == {r} and r not in {supertrial(t) for t in train}
        assert {t.user for t in test} == set(data.users)          # every surgeon is tested in every fold
        seen |= {t.name for t in test}
    assert seen == {t.name for t in data.trials}                   # each trial is tested exactly once
