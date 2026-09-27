"""Quick self-test: runs prepare -> train -> in-process bench on a tiny synthetic file.
Run:  python -m pytest python/tests -q     (or: python python/tests/test_pipeline.py)
Never report numbers from this test.
"""
import subprocess, sys, tempfile, shutil
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
ROOT = SRC.parents[1]


def run(*args):
    r = subprocess.run([sys.executable, *args], cwd=SRC, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_smoke():
    csv = ROOT / "data" / "raw" / "Base_synthetic.csv"
    try:
        run("s00_make_synthetic_baf.py", "--rows", "30000", "--out", str(csv))
        assert "train" in run("s02_prepare_data.py", "--csv", str(csv))
        assert "ONNX parity" in run("s03_train_sklearn.py")
        run("s05_bench_inprocess.py", "--n", "50")
        assert "median_us" in run("s07_stats.py", "--reference", "py_sklearn_njobs1")
    finally:
        csv.unlink(missing_ok=True)
        for sub in ("processed", ):
            for f in (ROOT / "data" / sub).glob("*"):
                if f.name != ".gitkeep": f.unlink()
        for sub in ("raw", "tables", "figures", "models"):
            for f in (ROOT / "results" / sub).glob("*"):
                if f.name != ".gitkeep": f.unlink()


if __name__ == "__main__":
    test_smoke(); print("smoke test passed")
