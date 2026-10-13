import numpy as np
import pytest
from src.protocol import (parse_events, protocol_params, assign_phases, map_emotions,
                          build_labels, label_session)
from src.io import sync_timestamps

GESTOS = ["Sonrisa", "Disgusto", "Sorprendido", "Triste"]


@pytest.fixture
def intervals(events):
    return map_emotions(assign_phases(parse_events(events), protocol_params(events)))


def test_phase_sequence(intervals):
    # 1 reposo inicial + 8 trials (2 reps x 4 gestos) x (Countdown, Gesto, Reposo)
    assert len(intervals) == 1 + 8 * 3
    assert list(intervals.phase[:4]) == ["Reposo", "Countdown", "Gesto", "Reposo"]
    assert list(intervals.loc[intervals.phase == "Gesto", "label"]) == GESTOS * 2


def test_durations_match_protocol(intervals):
    d = (intervals.t_end_ns - intervals.t_start_ns) / 1e9
    assert np.allclose(d[intervals.phase == "Gesto"], 3.0, atol=0.01)
    assert np.allclose(d[intervals.phase == "Countdown"], 3.0, atol=0.01)
    after = intervals.reason == "after_gesture"
    assert np.allclose(d[after], 5.0, atol=0.01)      # incluye el último, cerrado por sequence_end
    assert intervals.loc[after, "label"].eq("Reposo").all()   # no hereda el gesto previo


def test_no_overlaps(intervals):
    assert (intervals.t_start_ns.iloc[1:].to_numpy() >= intervals.t_end_ns.iloc[:-1].to_numpy()).all()


def test_sample_labels(events, emg, intervals):
    lb = build_labels(emg.t_ns, intervals)
    ev = lambda name: int(events.loc[events.event == name, "t_ns"].iloc[0])
    at = lambda ns: lb.iloc[np.searchsorted(lb.t_ns, ns)]
    assert lb.label.iloc[0] == "Fuera_protocolo"                      # antes del protocolo
    assert lb.label.iloc[-1] == "Fuera_protocolo"                     # después de sequence_end
    assert at(ev("gesture_show") + int(1e9)).label == "Countdown"
    assert at(ev("hold_start") + int(1e9)).label == "Sonrisa"
    assert at(ev("hold_start") + int(1e9)).trial == 0
    assert at(ev("hold_start") + int(4e9)).label == "Reposo"


def test_label_session_end_to_end(events, emg):
    s = sync_timestamps(emg, events, subject="s1", session="x")
    iv, lb = label_session(s)
    assert len(lb) == len(emg)
    assert set(lb.label) == {"Fuera_protocolo", "Reposo", "Countdown", "Sonrisa",
                             "Disgusto", "Sorprendido", "Triste"}
    assert 2900 <= (lb.label == "Sonrisa").sum() <= 3100   # 2 reps x 3 s x 500 Hz