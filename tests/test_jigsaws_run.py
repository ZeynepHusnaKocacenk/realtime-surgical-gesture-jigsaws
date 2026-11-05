"""JIGSAWS entry points, exercised on generated data in the real file layout (pipeline test only)."""
import shutil
from pathlib import Path

import pytest

from rtsurg.data import generate_synthetic_jigsaws, load_jigsaws_task
from rtsurg.jigsaws_check import check_task_dir, find_task_dir
import run_jigsaws


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = tmp_path_factory.mktemp("jig")
    generate_synthetic_jigsaws(r, task="Suturing", trials_per_user=2, seed=0)
    nested = r / "Needle_Passing"; nested.mkdir()
    generate_synthetic_jigsaws(nested, task="Needle_Passing", trials_per_user=2, seed=1)   # Needle_Passing/Needle_Passing
    return r


def test_find_and_check_task_dirs(root):
    assert find_task_dir(root, "Suturing") == root / "Suturing"
    assert find_task_dir(root, "Needle_Passing") == root / "Needle_Passing" / "Needle_Passing"
    assert find_task_dir(root, "Knot_Tying") is None
    assert check_task_dir(root / "Suturing") == []


def test_loader_tolerates_windows_line_endings_tabs_and_extra_files(root, tmp_path):
    d = tmp_path / "Suturing"; shutil.copytree(root / "Suturing", d)
    for f in list(d.rglob("*.txt")):
        f.write_text(f.read_text().replace("\n", "\r\n"))
    meta = next(d.glob("meta_file_*.txt"))
    meta.write_text(meta.read_text().replace(" ", "\t"))
    (d / "transcriptions" / "README.txt").write_text("not a trial\r\n")
    (d / "kinematics" / "AllGestures" / "Suturing_B999.txt").write_text("1 2 3\n")   # no transcription: ignored
    data = load_jigsaws_task(d)
    assert len(data.trials) == 16 and data.n_features == 38


def test_checker_flags_wrong_column_count(root, tmp_path):
    d = tmp_path / "Suturing"; shutil.copytree(root / "Suturing", d)
    f = next((d / "kinematics" / "AllGestures").glob("*.txt"))
    f.write_text("1 2 3\n4 5 6\n")
    assert any("columns" in p for p in check_task_dir(d))


def test_run_jigsaws_check_and_quick_run(root, tmp_path, capsys):
    assert run_jigsaws.main(["--root", str(root), "--check"]) == 0
    out = capsys.readouterr().out
    assert "[Knot_Tying] not found" in out and "Suturing: 16 trials" in out
    rc = run_jigsaws.main(["--root", str(root), "--tasks", "Suturing", "--quick", "--out-root", str(tmp_path)])
    assert rc == 0 and (tmp_path / "jigsaws_suturing_quick" / "results.md").exists()
    assert run_jigsaws.main(["--root", str(tmp_path / "empty"), "--check"]) == 2
