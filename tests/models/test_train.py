import json
import numpy as np
from src.models import cross_validate, run_experiment, load_model
from src.models.common import undersample_indices
from tests.toy import make_toy_df, small_cfg


def test_undersample_caps_majority_only_in_train():
    y = np.array([0] * 100 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10)
    keep = undersample_indices(y, seed=0)
    assert np.bincount(y[keep]).tolist() == [10, 10, 10, 10, 10]


def test_cross_validate_separable_and_no_leakage():
    df = make_toy_df(separable=True)
    pred = cross_validate(df, "rf", models_cfg=small_cfg())
    assert set(pred["rep"]) == {0, 1, 2, 3, 4}                      # el reposo inicial (rep=-1) nunca es test
    assert len(pred) == (df["rep"] >= 0).sum()
    assert (pred["y_true"] == pred["y_pred"]).mean() > 0.9


def test_run_experiment_writes_artifacts(tmp_path):
    df, cfg = make_toy_df(), small_cfg()
    r = run_experiment(df, "rf", models_cfg=cfg, out_root=tmp_path)
    for f in ("metrics.json", "predictions.csv", "confusion_matrix.png", "model.joblib"):
        assert (tmp_path / "rf" / f).stat().st_size > 0
    m = json.loads((tmp_path / "rf" / "metrics.json").read_text())
    assert m["meta"]["folds"] == ["rep=0", "rep=1", "rep=2", "rep=3", "rep=4"]
    assert m["pooled"]["macro_f1"] > 0.9
    model, meta = load_model(tmp_path / "rf" / "model.joblib")
    assert meta["model"] == "rf" and model.predict(df.filter(like="f").to_numpy()[:3]).shape == (3,)


def test_dummy_is_near_chance(tmp_path):
    r = run_experiment(make_toy_df(), "dummy", models_cfg=small_cfg(), out_root=tmp_path)
    assert r["summary"]["pooled"]["macro_f1"] < 0.5