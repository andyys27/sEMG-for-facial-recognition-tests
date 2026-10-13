"""Fixtures: primeras filas reales de Unity_events.csv + EMG sintético."""
import numpy as np
import pandas as pd
import pytest

UNITY_CSV = """Timestamp,t_ns,source,t_rx_ns,t_unity_ns,offset_ns,seq,label,detail
2026-09-24 19:44:49.288088200,1790279089288088200,rx,1790279089288088200,1790279089161984400,,1,session_start,anchor_unix_ns=1790279089153836200;stopwatch_hz=10000000;unity=6000.3.24f1;udp=10.43.42.208:5005
2026-09-24 19:44:49.753856550,1790279089753856550,unity+offset,1790279089755988900,1790279089632160500,121696050,2,sequence_start,reps=5;gestures=4;initial_rest_s=5;countdown_s=3;hold_s=3;rest_s=5
2026-09-24 19:44:49.754632150,1790279089754632150,unity+offset,1790279089757337500,1790279089632936100,121696050,3,rest_start,reason=initial;trial=-1;rep=-1;gesture_idx=-1;gesture=none
2026-09-24 19:44:55.788350850,1790279095788350850,unity+offset,1790279095790884200,1790279095666620500,121730350,4,gesture_show,trial=0;rep=0;gesture_idx=0;gesture=Sonrisa
2026-09-24 19:44:55.788849650,1790279095788849650,unity+offset,1790279095804866000,1790279095667119300,121730350,5,countdown_3,trial=0;rep=0;gesture_idx=0;gesture=Sonrisa
2026-09-24 19:44:58.792314750,1790279098792314750,unity+offset,1790279098794778400,1790279098670584400,121730350,8,hold_start,trial=0;rep=0;gesture_idx=0;gesture=Sonrisa
2026-09-24 19:45:01.793428950,1790279101793428950,unity+offset,1790279101795949800,1790279101671698600,121730350,9,rest_start,reason=after_gesture;trial=0;rep=0;gesture_idx=0;gesture=Sonrisa
"""

FS = 500
T0 = 1790279080_000_000_000   # 19:44:40 UTC
T1 = 1790279145_000_000_000   # 19:45:45 UTC


@pytest.fixture
def events(tmp_path):
    from src.io import load_events
    p = tmp_path / "Unity_events.csv"
    p.write_text(UNITY_CSV, encoding="utf-8")
    return load_events(p)


@pytest.fixture
def emg():
    t = np.arange(T0, T1, int(1e9 / FS), dtype="int64")
    df = pd.DataFrame(np.zeros((len(t), 4)), columns=["EMG1", "EMG2", "EMG3", "EMG4"])
    df.insert(0, "t_ns", t)
    return df