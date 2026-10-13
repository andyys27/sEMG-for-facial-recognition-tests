"""Fixtures: eventos de Unity sintéticos (mismas columnas que Unity_events.csv) + EMG sintético."""
import numpy as np
import pandas as pd
import pytest

FS = 500
T_START = 1790279089_288_088_200          # session_start (ns desde epoch)
GESTURES = ("Sonrisa", "Disgusto", "Sorprendido", "Triste")
HEADER = "Timestamp,t_ns,source,t_rx_ns,t_unity_ns,offset_ns,seq,label,detail"


def _ts(ns: int) -> str:
    return pd.Timestamp(ns).strftime("%Y-%m-%d %H:%M:%S") + f".{ns % 10**9:09d}"


def make_unity_csv(reps=2, gestures=GESTURES, initial_rest=6.0,
                   countdown=3.0, hold=3.0, rest=5.0) -> str:
    """Genera el texto de un Unity_events.csv con la misma secuencia que tu protocolo."""
    s = lambda x: int(x * 1e9)
    rows, seq = [], 0

    def add(t, label, detail):
        nonlocal seq
        seq += 1
        rows.append(f"{_ts(t)},{t},unity+offset,{t},{t},0,{seq},{label},{detail}")

    t = T_START
    add(t, "session_start", "unity=test")
    t += s(0.4)
    add(t, "sequence_start", f"reps={reps};gestures={len(gestures)};initial_rest_s=5;"
                             f"countdown_s={countdown:g};hold_s={hold:g};rest_s={rest:g}")
    add(t, "rest_start", "reason=initial;trial=-1;rep=-1;gesture_idx=-1;gesture=none")
    t += s(initial_rest)
    for rep in range(reps):
        for gi, g in enumerate(gestures):
            d = f"trial={rep * len(gestures) + gi};rep={rep};gesture_idx={gi};gesture={g}"
            add(t, "gesture_show", d)
            add(t + 500_000, "countdown_3", d)
            add(t + s(1), "countdown_2", d)
            add(t + s(2), "countdown_1", d)
            add(t + s(countdown), "hold_start", d)
            add(t + s(countdown + hold), "rest_start", "reason=after_gesture;" + d)
            t += s(countdown + hold + rest)
    add(t, "sequence_end", "")
    return "\n".join([HEADER, *rows]) + "\n"


@pytest.fixture
def events(tmp_path):
    from src.io import load_events
    p = tmp_path / "Unity_events.csv"
    p.write_text(make_unity_csv(), encoding="utf-8")
    return load_events(p)


@pytest.fixture
def emg(events):
    t = np.arange(events.t_ns.min() - int(9e9), events.t_ns.max() + int(5e9),
                  int(1e9 / FS), dtype="int64")
    df = pd.DataFrame(np.zeros((len(t), 4)), columns=["EMG1", "EMG2", "EMG3", "EMG4"])
    df.insert(0, "t_ns", t)
    return df


BURSTS = {"Sonrisa": [2, 3], "Disgusto": [0, 1], "Sorprendido": [0, 1], "Triste": []}   # canales activos


@pytest.fixture
def raw_session(tmp_path):
    """Sesión cruda en disco (1000 Hz): ruido de 10 uV y ráfagas x10 durante cada gesto."""
    sess = tmp_path / "raw" / "subject_01" / "session_01"
    sess.mkdir(parents=True)
    (sess / "Unity_events.csv").write_text(make_unity_csv(), encoding="utf-8")
    ev = pd.read_csv(sess / "Unity_events.csv")
    t = np.arange(ev.t_ns.min() - int(9e9), ev.t_ns.max() + int(5e9), int(1e6), dtype="int64")
    amp = np.full((len(t), 4), 1e-5)
    for _, r in ev[ev.label == "hold_start"].iterrows():
        g = r.detail.split("gesture=")[1]
        i0, i1 = np.searchsorted(t, [r.t_ns + int(0.3e9), r.t_ns + int(3.3e9)])
        for c in BURSTS[g]:
            amp[i0:i1, c] *= 10
    emg = pd.DataFrame(np.random.default_rng(0).normal(0, 1, (len(t), 4)) * amp,
                       columns=["EMG1", "EMG2", "EMG3", "EMG4"])
    emg.insert(0, "Timestamp", pd.to_datetime(t, unit="ns").strftime("%Y-%m-%d %H:%M:%S.%f"))
    emg.to_csv(sess / "FREEEMG_EMG_with_timestamp.csv", index=False)
    return sess