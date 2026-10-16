"""Segmenta una sesión y genera sus diagnósticos en data/analysis.

    python -m scripts.analyze_session data/raw/subject_01/session_01
    python -m scripts.analyze_session data/raw/subject_01/session_01 --print   # tabla de activación
"""
import argparse

from src.analysis import analyze_session
from src.pipeline.orchestration import process_session


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--print", action="store_true", dest="show", help="imprime el perfil de activación")
    a = ap.parse_args()

    res = process_session(a.session_dir)
    out = analyze_session(res, print_profile=a.show)
    rep = out["report"]

    print(f"Señal filtrada: {'reutilizada' if res['cached'] else 'recalculada'} -> {res['interim']}")
    print(f"Segmentos válidos: {rep['segments']['valid']}/{rep['segments']['total']}")
    print(f"Ventanas por etiqueta: {rep['windows_per_label']}")
    for w in rep["warnings"]:
        print(f"AVISO: {w}")
    print(f"Análisis guardado en {out['dir']}")


if __name__ == "__main__":
    main()