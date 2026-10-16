"""De sesiones crudas a dataset de features: segmenta, analiza, arma y exporta.

    python -m scripts.prepare_data
"""
from pathlib import Path
import pandas as pd

from src.analysis import analyze_session, write_dataset_report
from src.config import load_yaml
from src.dataset import build_dataset, export_dataset, feature_columns, get_splits
from src.pipeline.orchestration import process_session


def main() -> None:
    raw_dir = Path(load_yaml("experiment/settings.yaml")["paths"]["raw"])
    sessions = [d for s in sorted(raw_dir.glob("*")) if s.is_dir()
                for d in sorted(s.glob("*")) if (d / "Unity_events.csv").exists()]
    if not sessions:
        raise SystemExit(f"No hay sesiones con Unity_events.csv en {raw_dir}")

    for d in sessions:
        r = process_session(d)
        a = analyze_session(r)
        print(f"[{r['stem']}] {int(r['segments'].valid.sum())}/{len(r['segments'])} segmentos válidos, "
              f"{len(r['windows']['label'])} ventanas | filtrado {'reutilizado' if r['cached'] else 'calculado'}")
        for w in a["report"]["warnings"]:
            print(f"    AVISO: {w}")

    df, raw = build_dataset()
    folds = list(get_splits(df))
    p_csv, p_raw = export_dataset(df, raw)
    rep = write_dataset_report(df, folds)

    print(f"\nDataset: {df.shape[0]} ventanas x {len(feature_columns(df))} features")
    print("\nVentanas por sujeto y etiqueta:")
    print(pd.crosstab(df["subject"], df["label"]).to_string())
    print("\nFolds de evaluación:")
    for f in folds:
        cls = df.iloc[f.test]["label"].value_counts().to_dict()
        print(f"  {f.name:<12} train={len(f.train):>5}  test={len(f.test):>4}  test por clase={cls}")
    print(f"\nDataset:  {p_csv}\n          {p_raw}\nReportes: {rep['dir']}")


if __name__ == "__main__":
    main()