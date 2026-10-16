"""
Etiquetado del CSV completo
Cada muestra del dataframe original recibe:
  - Block:   numero de bloque/repeticion (0 si esta en reposo/no asignado)
  - Emotion: nombre de la emocion asociada (vacio si es reposo)
  - Phase:   "Reposo" | "Countdown" | "Gesto"
  - Label:   etiqueta final lista para usar como target de clasificacion
             ("Reposo", "Countdown_<Emocion>", o "<Emocion>")
 
Util para entrenar modelos que necesitan ver la señal continua completa
con su etiqueta correspondiente
"""

import numpy as np
from pathlib import Path


def build_labeled_dataframe(df, time, consensus_events, cfg):
    n = len(time)
    label = np.full(n, cfg.reposo_label, dtype=object)
    phase = np.full(n, "Reposo", dtype=object)
    block_arr = np.zeros(n, dtype=int)
    emotion_arr = np.full(n, "", dtype=object)

    emotion_cycle = cfg.emotion_cycle

    for idx, ev in enumerate(consensus_events):
        emotion = emotion_cycle[idx % len(emotion_cycle)]
        block = (idx // len(emotion_cycle)) + 1

        onset_t = ev["onset_t"]
        offset_t = ev["offset_t"]

        # Fase de countdown: [onset - countdown_sec, onset) 
        cd_start = onset_t - cfg.countdown_sec
        mask_cd = (time >= cd_start) & (time < onset_t)
        label[mask_cd] = f"{cfg.countdown_label_prefix}{emotion}"
        phase[mask_cd] = "Countdown"
        block_arr[mask_cd] = block
        emotion_arr[mask_cd] = emotion

        # Fase de gesto 
        t0 = onset_t - cfg.pre_margin_sec
        t1 = offset_t + cfg.post_margin_sec
        mask_g = (time >= t0) & (time <= t1)
        label[mask_g] = emotion
        phase[mask_g] = "Gesto"
        block_arr[mask_g] = block
        emotion_arr[mask_g] = emotion

    out = df.copy()
    out["Block"] = block_arr
    out["Emotion"] = emotion_arr
    out["Phase"] = phase
    out["Label"] = label
    return out


def export_labeled_csv(df_labeled, cfg):
    if not cfg.export_labeled_full_csv:
        return None

    root = Path(cfg.output_path)
    root.mkdir(parents=True, exist_ok=True)
    out_path = root / cfg.labeled_csv_name
    df_labeled.to_csv(out_path, index=False)

    n_reposo = int((df_labeled["Phase"] == "Reposo").sum())
    n_countdown = int((df_labeled["Phase"] == "Countdown").sum())
    n_gesto = int((df_labeled["Phase"] == "Gesto").sum())
    print(f"  -> {cfg.labeled_csv_name}  "
          f"({len(df_labeled):,} muestras: {n_reposo:,} reposo / "
          f"{n_countdown:,} countdown / {n_gesto:,} gesto)")
    return out_path