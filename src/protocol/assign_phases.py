"""Intervalos de fase (Reposo / Countdown / Gesto) a partir de las marcas."""
import pandas as pd
from ..config import load_yaml


def assign_phases(marks: pd.DataFrame, params: dict | None = None,
                  phases_cfg: dict | None = None, end_ns: int | None = None) -> pd.DataFrame:
    """Une marcas consecutivas de la misma fase+trial y cierra cada intervalo en la
    siguiente marca. Devuelve: t_start_ns, t_end_ns, phase, trial, rep, gesture_idx,
    gesture, reason.

    Si el último intervalo no tiene evento siguiente, usa la duración nominal de
    `params` (rest_s / countdown_s / hold_s) o, si falta, `end_ns`.
    """
    cfg = phases_cfg or load_yaml("protocol/phases.yaml")
    tail_key = cfg["tail_duration_param"]
    m = marks.sort_values("t_ns", kind="stable").reset_index(drop=True)

    same = (m["phase"].eq(m["phase"].shift()) & m["phase"].notna()
            & m["trial"].eq(m["trial"].shift()))
    gid = (~same).cumsum()
    firsts = m.groupby(gid, sort=True).head(1).reset_index(drop=True)

    rows = []
    for i, r in firsts.iterrows():
        if pd.isna(r["phase"]):
            continue
        if i + 1 < len(firsts):
            t_end = int(firsts.loc[i + 1, "t_ns"])
        else:
            dur = (params or {}).get(tail_key.get(r["phase"], ""), None)
            t_end = int(r["t_ns"] + dur * 1e9) if dur is not None else int(end_ns or r["t_ns"])
        rows.append({"t_start_ns": int(r["t_ns"]), "t_end_ns": t_end, "phase": r["phase"],
                     "trial": int(r["trial"]), "rep": int(r["rep"]),
                     "gesture_idx": int(r["gesture_idx"]), "gesture": r["gesture"],
                     "reason": r["reason"]})
    return pd.DataFrame(rows)