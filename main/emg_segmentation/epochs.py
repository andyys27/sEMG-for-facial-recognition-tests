"""
Extraccion de epocas individuales por gesto y exportacion a CSV
(un archivo por bloque/emocion, mas el reporte de metricas de deteccion).
"""

import numpy as np
import pandas as pd
from pathlib import Path


def extract_epochs(df, time, consensus_events, cfg):
    emotion_cycle = cfg.emotion_cycle
    epochs = []

    for idx, ev in enumerate(consensus_events):
        emotion = emotion_cycle[idx % len(emotion_cycle)]
        block = (idx // len(emotion_cycle)) + 1

        t_start = ev["onset_t"] - cfg.pre_margin_sec
        t_end = ev["offset_t"] + cfg.post_margin_sec
        mask = (time >= t_start) & (time <= t_end)

        if mask.sum() == 0:
            print(f"Evento {idx+1} ({emotion} B{block}): sin muestras "
                  f"en [{t_start:.2f}, {t_end:.2f}]s — omitido.")
            continue

        sub = df[mask].copy()
        sub["Block"] = block
        sub["Emotion"] = emotion
        sub["Epoch_Time"] = sub["Time_Seconds"].values - time[np.where(mask)[0][0]]
        sub["N_Groups"] = ev["n_groups"]
        sub["Active_Channels"] = ", ".join(sorted(set(ev["channels_active"])))
        epochs.append(sub)

    return epochs


def export_csvs(epochs, cfg):
    # CSV individual por bloque/gesto
    root = Path(cfg.output_path)
    for emo in cfg.emotion_cycle:
        (root / emo).mkdir(parents=True, exist_ok=True)

    for sub in epochs:
        emo = sub["Emotion"].iloc[0]
        block = int(sub["Block"].iloc[0])
        sub.to_csv(root / emo / f"bloque_{block:02d}.csv", index=False)


def export_metrics(consensus_events, cfg):
    # Reporte de deteccion exportado solo como CSV
    root = Path(cfg.output_path)
    outhpath = root / "Active_Channels"
    outhpath.mkdir(parents=True, exist_ok=True)

    emotion_cycle = cfg.emotion_cycle
    rows = []

    for idx, ev in enumerate(consensus_events):
        emo = emotion_cycle[idx % len(emotion_cycle)]
        block = (idx // len(emotion_cycle)) + 1
        rows.append({
            "Event_Index": idx + 1,
            "Block": block,
            "Emotion": emo,
            "Onset_s": round(ev["onset_t"], 4),
            "Offset_s": round(ev["offset_t"], 4),
            "Duration_s": round(ev["offset_t"] - ev["onset_t"], 4),
            "N_Groups": ev["n_groups"],
            "Groups_Active": ", ".join(sorted(ev["groups_active"])),
            "Channels_Active": ", ".join(sorted(set(ev["channels_active"]))),
        })

    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(outhpath / "detection_metrics.csv", index=False)

    print(f"\n  -> detection_metrics.csv   ({len(rows)} eventos)")
    return df_metrics