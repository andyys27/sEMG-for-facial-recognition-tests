import pytest
from src.evaluation.compare_models import write_comparison, pairwise_wilcoxon, load_runs
from src.evaluation.diagnose_generalization import diagnose_generalization, write_diagnosis
from src.models import run_experiment
from tests.toy import make_toy_df, small_cfg


@pytest.fixture
def runs_dir(tmp_path):
    df, cfg = make_toy_df(), small_cfg()
    for m in ("dummy", "rf", "lda"):
        run_experiment(df, m, models_cfg=cfg, out_root=tmp_path / "models", save_final=False)
    return tmp_path / "models"


def test_write_comparison(runs_dir, tmp_path):
    out = write_comparison(runs_dir, tmp_path / "reports")
    assert out["table"]["model"].iloc[-1] == "dummy"                 # el azar queda último
    for f in ("model_comparison.csv", "model_comparison.png", "model_comparison.md"):
        assert (tmp_path / "reports" / f).stat().st_size > 0
    assert "ninguna diferencia puede salir significativa" in out["summary"].read_text(encoding="utf-8")


def test_pairwise_min_p_with_5_folds(runs_dir):
    sig = pairwise_wilcoxon(load_runs(runs_dir))
    assert len(sig) == 3 and (sig["min_possible_p"] == 2 / 2 ** 5).all()


def test_diagnosis_detects_window_leakage(tmp_path):
    df = make_toy_df(separable=False)                               # solo se puede memorizar el segmento
    t = diagnose_generalization(df, ("rf", "knn"), small_cfg())
    assert (t["gap_macro_f1"] > 0.3).all()
    assert (t["honest_macro_f1"] < 0.4).all()
    assert write_diagnosis(t, "leave_one_rep_out", tmp_path).exists()