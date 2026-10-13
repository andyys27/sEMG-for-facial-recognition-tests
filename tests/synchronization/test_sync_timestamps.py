import pytest
from src.io import sync_timestamps, load_emg


def test_load_events_expands_detail(events):
    assert list(events["event"][:3]) == ["session_start", "sequence_start", "rest_start"]
    r = events.loc[events["event"] == "hold_start"].iloc[0]
    assert (r["trial"], r["rep"], r["gesture"]) == (0, 0, "Sonrisa")
    assert (events["t_ns"] - events["ts_ns"]).abs().max() < 1_000   # mismo instante


def test_sync_report(events, emg):
    s = sync_timestamps(emg, events, subject="s1", session="x")
    assert abs(s.report["emg"]["fs_est_hz"] - 500) < 1
    assert s.report["events"]["outside_emg"] == 0
    assert s.report["events"]["clock_skew_ms"] < 1


def test_non_monotonic_raises(events, emg):
    bad = emg.copy()
    bad.loc[10, "t_ns"] = bad.loc[5, "t_ns"]
    with pytest.raises(ValueError, match="no monotonos"):
        sync_timestamps(bad, events)


def test_no_overlap_raises(events, emg):
    shifted = emg.copy()
    shifted["t_ns"] += int(3600e9)        # EMG 1 h desfasado (zona horaria mal)
    with pytest.raises(ValueError, match="no se solapan"):
        sync_timestamps(shifted, events)


def test_load_emg_parses_nanoseconds(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("Timestamp,EMG1,EMG2,EMG3,EMG4\n"
                 "2026-09-24 19:44:40.176214784,0.0,0.0,0.0,0.0\n"
                 "2026-09-24 19:44:40.178214784,0.1,0.0,0.0,0.0\n")
    df = load_emg(p)
    assert df.t_ns.iloc[0] == 1790279080176214784
    assert df.t_ns.diff().iloc[1] == 2_000_000