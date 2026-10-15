import pytest
pytest.importorskip("keras")
from src.dataset import get_splits
from src.models import run_experiment
from tests.toy import make_toy_df, make_toy_raw, small_cfg


def test_cnn_runs_and_learns(tmp_path):
    df = make_toy_df(win_per_seg=12)                                  # una CNN necesita más ventanas que un RF
    folds = list(get_splits(df))[:2]                                  # 2 folds para que el test sea rápido
    r = run_experiment(df, "cnn", raw=make_toy_raw(df), folds=folds, models_cfg=small_cfg(), out_root=tmp_path)
    assert (tmp_path / "cnn" / "metrics.json").exists()
    assert not (tmp_path / "cnn" / "model.joblib").exists()           # la CNN no guarda modelo final (aún)
    assert len(r["pred"]) == df["rep"].isin([0, 1]).sum()
    assert r["summary"]["pooled"]["accuracy"] > 0.5                   # azar = 0.2