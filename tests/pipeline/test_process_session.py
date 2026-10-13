"""Prueba de integración: sesión cruda en disco -> segmentos -> dataset."""
import numpy as np
import pandas as pd
from tests.conftest import make_unity_csv
from src.pipeline.orchestration import process_session
from src.dataset import build_dataset, feature_columns


def test_raw_session_to_dataset(tmp_path):
    sess = tmp_path / "raw" / "subject_01" / "session_01"
    sess.mkdir(parents=True)
    (sess / "Unity_events.csv").write_text(make_unity_csv(), encoding="utf-8")

    ev = pd.read_csv(sess / "Unity_events.csv")
    t = np.arange(ev.t_ns.min() - int(9e9), ev.t_ns.max() + int(5e9), int(1e6), dtype="int64")  # 1000 Hz
    ts = pd.to_datetime(t, unit="ns").strftime("%Y-%m-%d %H:%M:%S.%f")
    rng = np.random.default_rng(0)
    emg = pd.DataFrame(rng.normal(0, 1e-5, (len(t), 4)), columns=["EMG1", "EMG2", "EMG3", "EMG4"])
    emg.insert(0, "Timestamp", ts)
    emg.to_csv(sess / "FREEEMG_EMG_with_timestamp.csv", index=False)

    out = tmp_path / "segments"
    r = process_session(sess, out)
    assert r["segments"].valid.all()
    assert (out / "subject_01_session_01_windows.npz").exists()

    df, raw = build_dataset(out)
    assert set(df.label) == {"Reposo", "Sonrisa", "Disgusto", "Sorprendido", "Triste"}
    assert len(feature_columns(df)) == 70
    assert df.subject.eq("subject_01").all()