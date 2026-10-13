import json
import numpy as np
from src.pipeline.orchestration import process_session
from src.analysis import analyze_session, write_dataset_report
from src.dataset import build_dataset, get_splits
from src.io.interim import cache_key


def _run(raw_session, tmp_path):
    return process_session(raw_session, tmp_path / "seg", interim_root=tmp_path / "interim",
                           labels_dir=tmp_path / "labels")


def test_interim_files_and_cache(raw_session, tmp_path):
    r1 = _run(raw_session, tmp_path)
    d = tmp_path / "interim" / "subject_01" / "session_01"
    for f in ("emg_filtered.npz", "events_parsed.csv", "sync_report.json"):
        assert (d / f).exists()
    assert not r1["cached"]
    r2 = _run(raw_session, tmp_path)
    assert r2["cached"]                                          # reutiliza la señal filtrada
    assert np.allclose(r1["filtered"]["EMG1"], r2["filtered"]["EMG1"])


def test_cache_key_changes_with_config():
    a = cache_key({"notch_hz": 60}, 1000.0, 100, 0)
    assert a == cache_key({"notch_hz": 60}, 1000.0, 100, 0)
    assert a != cache_key({"notch_hz": 50}, 1000.0, 100, 0)
    assert a != cache_key({"notch_hz": 60}, 1000.0, 101, 0)


def test_analyze_session_outputs(raw_session, tmp_path):
    out = analyze_session(_run(raw_session, tmp_path), analysis_root=tmp_path / "analysis")
    for f in ("overview.png", "activation_profile.png", "activation_profile.csv",
              "activity_by_label.png", "activity_by_label.csv", "quality_report.json"):
        assert (out["dir"] / f).stat().st_size > 0
    tab = out["activity"]
    assert tab.loc["Sonrisa", "EMG3"] > 3 * tab.loc["Reposo", "EMG3"]       # ráfaga en el canal correcto
    assert tab.loc["Disgusto", "EMG1"] > 3 * tab.loc["Reposo", "EMG1"]
    assert tab.loc["Triste", "EMG1"] < 2 * tab.loc["Reposo", "EMG1"]        # Triste sin activación
    rep = json.loads((out["dir"] / "quality_report.json").read_text())
    assert rep["segments"]["valid"] == rep["segments"]["total"] == 8 + 9    # 8 gestos + 9 reposos
    assert not any("Se esperaban" in w for w in rep["warnings"])               # 8 esperados, 8 encontrados


def test_dataset_report(raw_session, tmp_path):
    _run(raw_session, tmp_path)
    df, _ = build_dataset(tmp_path / "seg")
    out = write_dataset_report(df, list(get_splits(df)), tmp_path / "reports")
    assert "Ventanas" in out["summary"].read_text(encoding="utf-8")
    assert (tmp_path / "reports" / "class_balance.png").stat().st_size > 0
    assert (tmp_path / "reports" / "feature_class_heatmap.png").stat().st_size > 0