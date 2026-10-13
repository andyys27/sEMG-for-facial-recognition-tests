"""Reporte de calidad y gráficas de una sesión -> data/analysis/<sujeto>/<sesión>/."""
import json
from pathlib import Path
import pandas as pd

from ..config import load_yaml
from .signal_diagnostics import (plot_overview, activation_profile, plot_activation_profile,
                                 activity_by_label, plot_activity_by_label)

IMBALANCE_WARN = 3.0


def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


def quality_report(res: dict) -> dict:
    """Resumen verificable de la sesión: sincronía, conteos vs. protocolo y avisos."""
    s, iv, seg, params = res["session"], res["intervals"], res["segments"], res["params"]
    win = pd.Series(res["windows"]["label"]).value_counts()
    gest = [l for l in win.index if l != "Reposo"]
    imbalance = float(win.get("Reposo", 0) / win[gest].mean()) if gest else None

    warns = []
    expected = params.get("reps", 0) * params.get("gestures", 0)
    found = int((iv.phase == "Gesto").sum())
    if expected and found != expected:
        warns.append(f"Se esperaban {expected} gestos según sequence_start y se encontraron {found}")
    if not seg.valid.all():
        warns.append(f"{int((~seg.valid).sum())} segmentos descartados: {seg.loc[~seg.valid, 'reason'].tolist()}")
    if imbalance and imbalance > IMBALANCE_WARN:
        warns.append(f"Reposo tiene {imbalance:.1f}x más ventanas que el promedio de los gestos")
    if s.report["emg"]["gaps_gt_3x"]:
        warns.append(f"EMG con {s.report['emg']['gaps_gt_3x']} huecos")

    return {"subject": s.subject, "session": s.session, "sync": s.report, "protocol": params,
            "intervals_per_label": iv.groupby("label").size().to_dict(),
            "samples_per_label": res["labels"]["label"].value_counts().to_dict(),
            "windows_per_label": win.to_dict(),
            "segments": {"total": int(len(seg)), "valid": int(seg.valid.sum())},
            "reposo_vs_gesto_window_ratio": imbalance, "warnings": warns}


def analyze_session(res: dict, analysis_root: str | Path | None = None, print_profile: bool = False) -> dict:
    """Genera en data/analysis/<sujeto>/<sesión>/: overview.png, activation_profile.{png,csv},
    activity_by_label.{png,csv} y quality_report.json."""
    s = res["session"]
    root = analysis_root or load_yaml("experiment/settings.yaml")["paths"]["analysis"]
    out = Path(root) / s.subject / s.session
    out.mkdir(parents=True, exist_ok=True)
    ch, fs, filt = res["channels"], res["fs"], res["filtered"]

    plot_overview(filt, res["intervals"], res["segments"], ch, fs, out / "overview.png")

    prof = activation_profile(filt, s.events, ch, fs)
    prof.to_csv(out / "activation_profile.csv", index=False)
    plot_activation_profile(prof, res["margins"], res["params"], out / "activation_profile.png")
    if print_profile:
        print(prof[prof.gesture == "todos"].pivot(index="t_from", columns="anchor", values="activation").round(2))

    tab = activity_by_label(filt, res["labels"], ch, fs)
    tab.to_csv(out / "activity_by_label.csv")
    plot_activity_by_label(tab, out / "activity_by_label.png")

    rep = quality_report(res)
    (out / "quality_report.json").write_text(json.dumps(rep, indent=2, default=_jsonable), encoding="utf-8")
    return {"dir": out, "report": rep, "profile": prof, "activity": tab}