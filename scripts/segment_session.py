"""Segmenta una sesión y guarda ventanas etiquetadas.

    python -m scripts.segment_session data/raw/subject_01/session_01
"""
import argparse
import pandas as pd
from src.pipeline.orchestration import process_session


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--out", default="data/processed/segments")
    r = process_session(**{"session_dir": (a := ap.parse_args()).session_dir, "out_dir": a.out})

    seg, w = r["segments"], r["windows"]
    print(f"fs usada: {r['fs']:.1f} Hz")
    print("\nSegmentos (válidos / total) y duración media por etiqueta:")
    print(seg.assign(dur_s=seg.n_samples / r["fs"]).groupby("label")
          .agg(validos=("valid", "sum"), total=("valid", "size"), dur_s=("dur_s", "mean")).round(2))
    if (~seg.valid).any():
        print("\nSegmentos descartados:\n", seg.loc[~seg.valid, ["segment_id", "label", "reason"]])
    print("\nVentanas por etiqueta:")
    print(pd.Series(w["label"]).value_counts().to_string())
    print(f"\nForma de ventanas: {w['windows'].shape}  (n, muestras, canales)")
    print(f"Guardado en {r['dir']}/{r['stem']}_windows.npz")


if __name__ == "__main__":
    main()