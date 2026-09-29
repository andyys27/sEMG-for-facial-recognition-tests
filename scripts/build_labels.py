from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.labeling.build_labels import build_labels_from_unity, export_labels_csv


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    unity_csv = root / "data" / "raw" / "subject_01" / "session_01" / "Unity_events.csv"
    output_csv = root / "data" / "interim" / "labels" / "subject_01_session_01_labels.csv"

    labels = build_labels_from_unity(unity_csv, subject_id="subject_01", session_id="session_01")
    export_labels_csv(labels, output_csv)
    print(f"Labels generadas: {len(labels)} rows -> {output_csv}")


if __name__ == "__main__":
    main()
