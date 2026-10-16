"""Etiqueta una sesión desde Unity y muestra un resumen de verificación.

    python -m scripts.label_session data/raw/subject_01/session_01
"""
import argparse
from src.io import load_session
from src.protocol import label_session, export_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--out", default="data/processed/labels")
    a = ap.parse_args()

    s = load_session(a.session_dir)
    iv, lb = label_session(s)
    p_iv, p_lb = export_labels(s, iv, lb, a.out)

    r = s.report
    print(f"EMG: {r['emg']['n']} muestras, fs~{r['emg']['fs_est_hz']:.1f} Hz, "
          f"{r['emg']['duration_s']:.1f} s, huecos={r['emg']['gaps_gt_3x']}")
    print(f"Eventos Unity: {r['events']['n']}, desfase={r['events']['clock_skew_ms']:.3f} ms")
    iv = iv.assign(dur_s=(iv.t_end_ns - iv.t_start_ns) / 1e9)
    print("\nIntervalos por etiqueta (n, duración media s):")
    print(iv.groupby("label")["dur_s"].agg(["count", "mean"]).round(2).to_string())
    print("\nMuestras por etiqueta:")
    print(lb["label"].value_counts().to_string())
    print(f"\nGuardado: {p_iv}\n          {p_lb}")


if __name__ == "__main__":
    main()