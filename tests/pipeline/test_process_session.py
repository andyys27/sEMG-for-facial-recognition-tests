"""Prueba de integración: sesión cruda en disco -> labels/interim/segmentos -> dataset."""
from src.pipeline.orchestration import process_session
from src.dataset import build_dataset, feature_columns


def test_raw_session_to_dataset(raw_session, tmp_path):
    out = tmp_path / "segments"
    r = process_session(raw_session, out, interim_root=tmp_path / "interim", labels_dir=tmp_path / "labels")
    assert r["segments"].valid.all()
    assert (out / "subject_01_session_01_windows.npz").exists()
    assert (tmp_path / "labels" / "subject_01_session_01_labels.csv").exists()

    df, raw = build_dataset(out)
    assert set(df.label) == {"Reposo", "Sonrisa", "Disgusto", "Sorprendido", "Triste"}
    assert len(feature_columns(df)) == 70
    assert df.subject.eq("subject_01").all()