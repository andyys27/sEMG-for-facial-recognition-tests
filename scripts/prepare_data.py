"""De sesiones crudas a dataset de features: segmenta todas las sesiones, arma y exporta.

    python -m scripts.prepare_data
"""
from pathlib import Path
import pandas as pd

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
        print(f"[segmentación] {r['stem']}: {int(r['segments'].valid.sum())}/{len(r['segments'])} "
              f"segmentos válidos, {len(r['windows']['label'])} ventanas")

    df, raw = build_dataset()
    p_csv, p_raw = export_dataset(df, raw)
    print(f"\nDataset: {df.shape[0]} ventanas x {len(feature_columns(df))} features")
    print("\nVentanas por sujeto y etiqueta:")
    print(pd.crosstab(df["subject"], df["label"]).to_string())

    print("\nFolds de evaluación (ventanas de entrenamiento / prueba):")
    for f in get_splits(df):
        cls = df.iloc[f.test]["label"].value_counts().to_dict()
        print(f"  {f.name:<12} train={len(f.train):>5}  test={len(f.test):>4}  test por clase={cls}")
    print(f"\nGuardado:\n  {p_csv}\n  {p_raw}")


if __name__ == "__main__":
    main()