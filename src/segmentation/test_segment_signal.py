import numpy as np
import pytest
from src.protocol import parse_events, protocol_params, assign_phases, map_emotions
from src.segmentation import (segment_signal, validate_segments, temporal_windowing,
                              windows_from_segments, preprocess_emg)

CH = ["EMG1", "EMG2", "EMG3", "EMG4"]
MARGINS = {"Gesto": {"onset_s": 0.5, "offset_s": 0.0}, "Reposo": {"onset_s": 1.5, "offset_s": 0.5}}


@pytest.fixture
def intervals(events):
    return map_emotions(assign_phases(parse_events(events), protocol_params(events)))


@pytest.fixture
def noisy_emg(emg):
    rng = np.random.default_rng(0)
    out = emg.copy()
    out[CH] = rng.normal(0, 1e-5, (len(out), 4))
    return out


def test_margins_and_excluded_phases(noisy_emg, intervals):
    segs = segment_signal(noisy_emg, intervals, MARGINS, CH)
    assert all(s.phase in ("Gesto", "Reposo") for s in segs)          # sin Countdown
    gest = [s.duration_s for s in segs if s.phase == "Gesto"]
    assert np.allclose(gest, 2.5, atol=0.01)                          # 3 s - 0.5 s de onset
    rest = [s.duration_s for s in segs if s.phase == "Reposo"]
    assert np.isclose(max(rest), 4.0, atol=0.01)                      # reposo inicial 6 - 1.5 - 0.5
    assert np.isclose(min(rest), 3.0, atol=0.01)                      # post-gesto 5 - 1.5 - 0.5


def test_validation_flags_flatline(emg, intervals):
    segs = validate_segments(segment_signal(emg, intervals, MARGINS, CH), 500, 0.5)
    assert all((not s.valid) and s.reason == "flatline" for s in segs)


def test_validation_passes_on_noise(noisy_emg, intervals):
    segs = validate_segments(segment_signal(noisy_emg, intervals, MARGINS, CH), 500, 0.5)
    assert all(s.valid for s in segs)


def test_windowing_shape():
    w = temporal_windowing(np.zeros((3000, 4)), fs_hz=1000, window_ms=200, overlap=0.5)
    assert w.shape == (29, 200, 4)
    assert temporal_windowing(np.zeros((100, 4)), 1000, 200, 0.5).shape == (0, 200, 4)


def test_windows_inherit_labels(noisy_emg, intervals):
    segs = validate_segments(segment_signal(noisy_emg, intervals, MARGINS, CH), 500, 0.5)
    d = windows_from_segments(segs, fs_hz=500, window_ms=200, overlap=0.5)
    assert d["windows"].shape[0] == len(d["label"]) == len(d["trial"])
    assert set(d["label"]) == {"Reposo", "Sonrisa", "Disgusto", "Sorprendido", "Triste"}


def test_preprocess_removes_dc_and_mains():
    fs = 1000
    t = np.arange(0, 5, 1 / fs)
    import pandas as pd
    df = pd.DataFrame({"t_ns": (t * 1e9).astype("int64"),
                       "EMG1": 5.0 + np.sin(2 * np.pi * 60 * t) + np.sin(2 * np.pi * 100 * t)})
    cfg = {"remove_dc": True, "notch_hz": 60, "notch_q": 30, "bandpass_hz": [20, 450], "bandpass_order": 4}
    y = preprocess_emg(df, fs, cfg, ["EMG1"])["EMG1"].to_numpy()[500:-500]
    spec = np.abs(np.fft.rfft(y)) / len(y) * 2
    f = np.fft.rfftfreq(len(y), 1 / fs)
    assert abs(y.mean()) < 0.05
    assert spec[np.argmin(abs(f - 60))] < 0.05                         # 60 Hz atenuado
    assert spec[np.argmin(abs(f - 100))] > 0.8                         # 100 Hz conservado