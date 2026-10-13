"""Activación EMG promedio alineada a los eventos de Unity, para elegir márgenes.

    python -m scripts.inspect_margins data/raw/subject_01/session_01
"""
import sys
import numpy as np
from scipy.ndimage import uniform_filter1d
from src.config import load_yaml
from src.io import load_session
from src.segmentation import preprocess_emg


def main(session_dir: str) -> None:
    s = load_session(session_dir)
    cfg = load_yaml("experiment/settings.yaml")
    ch, fs = cfg["channel_names"], s.report["emg"]["fs_est_hz"]
    x = preprocess_emg(s.emg, fs, cfg["preprocessing"], ch)[ch].to_numpy()

    # Envolvente RMS (100 ms), normalizada por la mediana de cada canal (≈ nivel de reposo)
    env = np.sqrt(uniform_filter1d(x ** 2, size=int(0.1 * fs), axis=0, mode="nearest"))
    env = env / np.median(env, axis=0)
    act = env.max(axis=1)                       # canal más activo en cada instante
    t = s.emg["t_ns"].to_numpy()

    ev = s.events
    anchors = {"hold_start": ev.loc[ev.event == "hold_start", "t_ns"].to_numpy(),
               "reposo_post_gesto": ev.loc[(ev.event == "rest_start") &
                                           (ev.reason == "after_gesture"), "t_ns"].to_numpy()}
    edges = np.arange(-1.0, 5.01, 0.25)
    print("Activación = envolvente RMS / nivel de reposo (1.0 ≈ reposo), promedio de todos los trials\n")
    print(f"{'t_rel (s)':>9} | {'tras hold_start':>15} | {'tras rest_start':>15}")
    for a, b in zip(edges[:-1], edges[1:]):
        row = []
        for ts in anchors.values():
            vals = []
            for t0 in ts:
                i0, i1 = np.searchsorted(t, [t0 + a * 1e9, t0 + b * 1e9])
                if i1 > i0:
                    vals.append(act[i0:i1].mean())
            row.append(np.mean(vals))
        print(f"{a:>9.2f} | {row[0]:>15.2f} | {row[1]:>15.2f}")


if __name__ == "__main__":
    main(sys.argv[1])