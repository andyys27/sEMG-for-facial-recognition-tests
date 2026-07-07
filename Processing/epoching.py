import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# Configuracion
@dataclass
class Config:
    # Rutas
    csv_path: str = "../Test1/Data/FREEEMG_Processed_Signals.csv"
    output_path: str = "../Test1"

    # Protocolo
    num_blocks: int = 5
    emotion_cycle: list[str] = field(default_factory=lambda: [
        "Sonrisa", "Disgusto", "Sorprendido", "Triste"])

    # Grupos musculares
    channel_groups: dict[str, list[str]] = field(default_factory=lambda: {
        "Grupo_A": ["EMG1_Envelope_RMS", "EMG4_Envelope_RMS"],
        "Grupo_B":  ["EMG2_Envelope_RMS", "EMG3_Envelope_RMS"],
    })

    # Minimo de grupos que deben coincidir para validar un evento consenso
    min_groups_active: int = 1

    # Baseline y umbral
    baseline_window_sec: float = 20.0          # usado solo si use_rolling_baseline=False
    k_baseline: float = 3.0
    k_baseline_per_group: dict[str, float] = field(default_factory=dict)

    # --- Baseline movil robusto (mediana + MAD) ---
    use_rolling_baseline: bool = True
    rolling_baseline_window_sec: float = 20.0   # ventana causal para mediana/MAD
    rolling_baseline_min_frac: float = 0.25     # min_periods como fraccion de la ventana

    # --- Umbral doble (hysteresis / Schmitt trigger) ---
    # umbral_offset = baseline + (k_onset * k_offset_ratio) * sigma  (k_offset_ratio < 1)
    k_offset_ratio: float = 0.6
    k_offset_ratio_per_group: dict[str, float] = field(default_factory=dict)

    # --- Refinamiento con baseline local pre-countdown ---
    use_local_baseline_refinement: bool = True
    countdown_sec: float = 3.0          # duracion del countdown antes del gesto
    local_baseline_sec: float = 5.0     # ventana de reposo previa al countdown a usar
    local_refine_search_margin_sec: float = 2.0

    # --- Filtros de forma (energia y factor de cresta) ---
    use_shape_filters: bool = True
    energy_ratio_min: float = 3.0       # energia_evento / energia_ruido_baseline minima
    crest_factor_min: float = 1.05      # rechaza mesetas casi planas (probable ruido)
    crest_factor_max: float = 8.0       # rechaza picos puntiagudos (probable artefacto)

    # Rango de interes
    t_start: float = 0.0
    t_end: float = 380.0

    # Suavizado del envelope 
    smoothing_window_sec: float = 0.5

    # Separación minima entre eventos consecutivos
    min_inter_event_sec: float = 10.0

    # Tolerancia temporal para asociar eventos de diferentes grupos/canales
    tolerance_sec: float = 3.0

    # Gap filling
    gap_fill_sec: float = 1.5

    # Duración valida de un evento por canal 
    min_event_dur_sec: float = 2.5
    max_event_dur_sec: float = 10.0

    # Margenes al extraer la época
    pre_margin_sec:  float = 1.0
    post_margin_sec: float = 1.0

    # DPI gráficas
    plot_dpi: int = 200


# Utilidades de señal
def compute_threshold(signal, time, baseline_window_sec, k):
    # Umbral fijo (legado): mean + k·std sobre ventana de reposo inicial unica
    mask = time <= baseline_window_sec
    baseline = signal[mask]
    return float(baseline.mean() + k * baseline.std())


def smooth_signal(signal, fs, window_sec):
    # Media movil centrada
    w = max(1, int(window_sec * fs))
    return pd.Series(signal).rolling(window=w, center=True).mean().ffill().bfill().values


def rolling_baseline_stats(signal, fs, window_sec, min_frac=0.25):
    """
    Baseline movil robusto: mediana + MAD (Median Absolute Deviation) en
    ventana causal (solo pasado). La mediana/MAD toleran hasta ~50% de
    contaminacion por activaciones dentro de la ventana, por lo que no
    hace falta excluir manualmente los tramos de gesto: el propio
    estimador robusto los ignora en la práctica mientras sean minoria
    dentro de la ventana. Esto resuelve el drift lento (sudor, impedancia,
    tension residual) sin caer en el circulo de "necesito saber donde
    esta el reposo para calcular el reposo".
    """
    w = max(3, int(window_sec * fs))
    min_periods = max(3, int(w * min_frac))
    s = pd.Series(signal)

    med = s.rolling(window=w, min_periods=min_periods).median()
    mad = (s - med).abs().rolling(window=w, min_periods=min_periods).median()

    med = med.bfill().ffill().values
    mad = mad.bfill().ffill().values

    sigma = mad * 1.4826  # escala MAD -> equivalente a std bajo normalidad
    sigma = np.where(sigma < 1e-12, 1e-12, sigma)
    return med, sigma


def hysteresis_thresholds(med, sigma, k_on, k_off_ratio):
    # Umbral de encendido (estricto) y de apagado (mas laxo) -> evita flicker
    upper = med + k_on * sigma
    lower = med + (k_on * k_off_ratio) * sigma
    return upper, lower


def event_shape_metrics(signal_seg, sigma_seg, baseline_med_seg):
    """
    Metricas de forma para distinguir un gesto real de un artefacto:
      - energy_ratio: energia de la señal por encima del baseline, respecto
        a la energia del ruido de baseline en la misma ventana. Un pico
        breve que apenas cruza el umbral pero decae enseguida tiene poca
        energia real y se descarta.
      - crest_factor: pico / RMS del segmento. Un artefacto puntual
        (ej. "pop" de electrodo, movimiento de cable) tiene un crest factor
        muy alto (pico aislado, resto plano). Un gesto real, al ya venir
        suavizado como envelope, muestra una meseta sostenida con crest
        factor moderado. Un crest factor casi 1 (demasiado plano) sugiere
        que en realidad es solo fluctuacion de ruido alrededor del umbral.
    """
    excess = np.clip(signal_seg - baseline_med_seg, 0, None)
    energy_event = float(np.sum(excess ** 2))
    energy_noise_ref = float(np.sum(sigma_seg ** 2)) + 1e-12
    energy_ratio = energy_event / energy_noise_ref

    peak = float(signal_seg.max())
    rms = float(np.sqrt(np.mean(signal_seg ** 2))) + 1e-12
    crest_factor = peak / rms

    return energy_ratio, crest_factor


# Deteccion por canal individual
def detect_channel_events(signal, time, upper, lower, baseline_med, baseline_sigma, fs, cfg):
    """
    upper/lower/baseline_med/baseline_sigma: arrays del mismo largo que signal
    (umbral doble/hysteresis: se enciende con `upper`, se apaga con `lower`).
    """
    sig_smooth = smooth_signal(signal, fs, cfg.smoothing_window_sec)
    roi = (time >= cfg.t_start) & (time <= cfg.t_end)   # Region de interes

    # 1. Deteccion con hysteresis: enciende al cruzar upper, apaga al cruzar lower
    raw_events = []
    in_event = False
    onset_idx = None

    for i in range(1, len(sig_smooth)):
        if not roi[i]:
            if in_event:
                in_event  = False
                onset_idx = None
            continue

        rising = sig_smooth[i] > upper[i] and sig_smooth[i - 1] <= upper[i - 1]
        falling = sig_smooth[i] < lower[i] and sig_smooth[i - 1] >= lower[i - 1]

        if not in_event and rising:
            in_event  = True
            onset_idx = i
        elif in_event and falling:
            raw_events.append((onset_idx, i - 1))
            in_event  = False
            onset_idx = None

    if not raw_events:
        return []

    # 2. Gap filling
    gap_samples = cfg.gap_fill_sec * fs
    merged = [list(raw_events[0])]

    for (on, off) in raw_events[1:]:
        gap = on - merged[-1][1]
        if gap <= gap_samples:
            merged[-1][1] = off          # Extender el evento actual
        else:
            merged.append([on, off])

    # 3. Filtro de duracion, filtros de forma y calculo de amplitudes
    events = []
    for (on, off) in merged:
        dur = time[off] - time[on]
        if not (cfg.min_event_dur_sec <= dur <= cfg.max_event_dur_sec):
            continue

        seg = signal[on:off + 1]

        if cfg.use_shape_filters:
            energy_ratio, crest_factor = event_shape_metrics(
                seg, baseline_sigma[on:off + 1], baseline_med[on:off + 1])
            ok_energy = energy_ratio >= cfg.energy_ratio_min
            ok_crest  = cfg.crest_factor_min <= crest_factor <= cfg.crest_factor_max
            if not (ok_energy and ok_crest):
                continue
        else:
            energy_ratio, crest_factor = None, None

        events.append({
            "onset_idx": on,
            "offset_idx": off,
            "onset_t": float(time[on]),
            "offset_t": float(time[off]),
            "peak_amp": float(seg.max()),
            "mean_amp": float(seg.mean()),
            "energy_ratio": energy_ratio,
            "crest_factor": crest_factor,
        })

    if len(events) <= 1:
        return events

    # 4. Deduplicacion por distancia minima
    filtered = [events[0]]
    for ev in events[1:]:
        prev = filtered[-1]
        if ev["onset_t"] - prev["onset_t"] >= cfg.min_inter_event_sec:
            filtered.append(ev)
        elif ev["peak_amp"] > prev["peak_amp"]:
            filtered[-1] = ev

    return filtered


# Refinamiento con baseline local (5s previos al countdown de cada gesto)
def refine_event_local_baseline(signal, time, fs, onset_t, offset_t, cfg, k_on, k_off_ratio):
    """
    Redefine onset/offset de UN evento usando como baseline los
    `local_baseline_sec` segundos de reposo justo antes del countdown que
    precede al gesto (se asume que el countdown dura `countdown_sec` y
    ocurre inmediatamente antes del onset detectado en la pasada global).
    Esto captura el nivel de reposo especifico de ESE gesto (que puede
    variar respecto al reposo inicial por fatiga o tension residual de la
    emocion anterior), en vez de depender de un baseline global o de una
    ventana movil generica.
    Si no hay suficientes muestras de reposo disponibles, devuelve el
    evento original sin modificar.
    """
    reposo_end   = onset_t - cfg.countdown_sec
    reposo_start = reposo_end - cfg.local_baseline_sec
    mask_baseline = (time >= reposo_start) & (time < reposo_end)

    if mask_baseline.sum() < max(5, int(fs)):
        return onset_t, offset_t, None, None

    baseline_seg = signal[mask_baseline]
    med_local = float(np.median(baseline_seg))
    mad_local = float(np.median(np.abs(baseline_seg - med_local))) * 1.4826
    mad_local = max(mad_local, 1e-12)

    upper = med_local + k_on * mad_local
    lower = med_local + (k_on * k_off_ratio) * mad_local

    # Buscar el cruce real en una ventana amplia alrededor del evento original,
    # arrancando justo al terminar el reposo local (fin de countdown)
    search_start = reposo_end
    search_end   = offset_t + cfg.local_refine_search_margin_sec
    mask_search  = (time >= search_start) & (time <= search_end)
    idxs = np.where(mask_search)[0]
    if len(idxs) < 3:
        return onset_t, offset_t, med_local, mad_local

    sig_smooth = smooth_signal(signal[idxs], fs, cfg.smoothing_window_sec)

    new_onset_t, new_offset_t = onset_t, offset_t
    in_event = False
    found_onset = False
    for k in range(1, len(sig_smooth)):
        if not in_event and sig_smooth[k] > upper and sig_smooth[k - 1] <= upper:
            in_event = True
            found_onset = True
            new_onset_t = float(time[idxs[k]])
        elif in_event and sig_smooth[k] < lower:
            new_offset_t = float(time[idxs[k]])
            break

    if not found_onset:
        # No se encontro cruce claro con el baseline local: conservar el original
        return onset_t, offset_t, med_local, mad_local

    return new_onset_t, new_offset_t, med_local, mad_local


def refine_consensus_events(consensus_events, df, time, fs, cfg):
    """Aplica el refinamiento local a cada evento de consenso, canal por canal
    dentro de cfg.channel_groups, y combina el resultado (min onset, max offset)
    para mantener robustez ante que un canal no tenga baseline local valido."""
    if not cfg.use_local_baseline_refinement:
        return consensus_events

    all_channels = [ch for chs in cfg.channel_groups.values() for ch in chs]
    ch_to_k, ch_to_koff = {}, {}
    for gname, chs in cfg.channel_groups.items():
        k = cfg.k_baseline_per_group.get(gname, cfg.k_baseline)
        koff = cfg.k_offset_ratio_per_group.get(gname, cfg.k_offset_ratio)
        for ch in chs:
            ch_to_k[ch] = k
            ch_to_koff[ch] = koff

    refined = []
    for ev in consensus_events:
        onsets, offsets = [], []
        for ch in all_channels:
            new_on, new_off, _, _ = refine_event_local_baseline(
                df[ch].values, time, fs,
                ev["onset_t"], ev["offset_t"], cfg,
                ch_to_k[ch], ch_to_koff[ch])
            onsets.append(new_on)
            offsets.append(new_off)

        ev_ref = dict(ev)
        ev_ref["onset_t"]  = float(np.min(onsets))
        ev_ref["offset_t"] = float(np.max(offsets))
        refined.append(ev_ref)

    refined.sort(key=lambda x: x["onset_t"])
    return refined


# Consenso por grupos musculares
def detect_group_events(per_channel_events, cfg):
    group_events = {}

    for group_name, ch_list in cfg.channel_groups.items():
        # Recopilar todos los eventos de los canales del grupo
        all_ev = []
        for ch in ch_list:
            for ev in per_channel_events.get(ch, []):
                all_ev.append((ev["onset_t"], ch, ev))
        all_ev.sort(key=lambda x: x[0])

        if not all_ev:
            group_events[group_name] = []
            continue

        # Agrupar eventos solapados del mismo grupo por ventana de tolerancia
        merged = []
        used = set()

        for i, (t_i, ch_i, ev_i) in enumerate(all_ev):
            if i in used:
                continue
            cluster = [(ch_i, ev_i)]
            used.add(i)

            for j, (t_j, ch_j, ev_j) in enumerate(all_ev):
                if j in used or ch_j == ch_i:
                    continue
                if abs(t_j - t_i) <= cfg.tolerance_sec:
                    cluster.append((ch_j, ev_j))
                    used.add(j)

            g_onset  = min(e["onset_t"]  for _, e in cluster)
            g_offset = max(e["offset_t"] for _, e in cluster)
            g_peak   = max(e["peak_amp"] for _, e in cluster)
            merged.append({
                "onset_t":       g_onset,
                "offset_t":      g_offset,
                "peak_amp":      g_peak,
                "channels_active": [ch for ch, _ in cluster],
            })

        # Deduplicar por distancia minima dentro del grupo
        merged.sort(key=lambda x: x["onset_t"])
        dedup = [merged[0]]
        for ev in merged[1:]:
            if ev["onset_t"] - dedup[-1]["onset_t"] >= cfg.min_inter_event_sec:
                dedup.append(ev)
            elif ev["peak_amp"] > dedup[-1]["peak_amp"]:
                dedup[-1] = ev

        group_events[group_name] = dedup

    return group_events


def build_consensus_events(group_events, cfg):
    group_names = list(group_events.keys())

    # Pool de todos los eventos de todos los grupos
    pool = []
    for gname in group_names:
        for ev in group_events[gname]:
            pool.append((ev["onset_t"], gname, ev))
    pool.sort(key=lambda x: x[0])

    used = set()
    consensus = []

    for i, (t_i, g_seed, ev_seed) in enumerate(pool):
        if i in used:
            continue

        cluster = [(g_seed, ev_seed)]
        used.add(i)

        for j, (t_j, g_j, ev_j) in enumerate(pool):
            if j in used or g_j == g_seed:
                continue
            if abs(t_j - t_i) <= cfg.tolerance_sec:
                cluster.append((g_j, ev_j))
                used.add(j)

        active_groups = list({g for g, _ in cluster})
        if len(active_groups) < cfg.min_groups_active:
            continue

        c_onset  = min(e["onset_t"]  for _, e in cluster)
        c_offset = max(e["offset_t"] for _, e in cluster)

        consensus.append({
            "onset_t":       c_onset,
            "offset_t":      c_offset,
            "groups_active": active_groups,
            "n_groups":      len(active_groups),
            "channels_active": [
                ch for _, ev in cluster
                for ch in ev["channels_active"]
            ],
        })

    # Ordenar y deduplicar solapamientos residuales
    consensus.sort(key=lambda x: x["onset_t"])
    if not consensus:
        return []

    clean = [consensus[0]]
    for ev in consensus[1:]:
        if ev["onset_t"] - clean[-1]["onset_t"] >= cfg.min_inter_event_sec:
            clean.append(ev)

    return clean


# Extracción de epocas
def extract_epochs(df, time, consensus_events, cfg):
    emotion_cycle = cfg.emotion_cycle
    epochs = []

    for idx, ev in enumerate(consensus_events):
        emotion = emotion_cycle[idx % len(emotion_cycle)]
        block = (idx // len(emotion_cycle)) + 1

        t_start = ev["onset_t"]  - cfg.pre_margin_sec
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


# Exportación
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
    outhpath = root/"Active_Channels"
    
    outhpath.mkdir(parents=True, exist_ok=True)

    emotion_cycle = cfg.emotion_cycle
    rows = []

    for idx, ev in enumerate(consensus_events):
        emo   = emotion_cycle[idx % len(emotion_cycle)]
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

    # Anadir fila de thresholds al pie como metadatos legibles
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(outhpath / "detection_metrics.csv", index=False)

    print(f"\n  -> detection_metrics.csv   ({len(rows)} eventos)")
    return df_metrics


# Graficas
EMOTION_COLORS = {
    "Sonrisa":"green",
    "Disgusto":"red",
    "Sorprendido":"blue",
    "Triste":"purple",
}

# Colores por grupo muscular para graficas individuales
GROUP_COLORS = {
    "Grupo_A":"orange",
    "Grupo_B":"teal",
}


def plot_per_channel(df, time, per_channel_events, thresholds, consensus_events, cfg):
    root = Path(cfg.output_path)
    outhpath = root/"Active_Channels"
    
    outhpath.mkdir(parents=True, exist_ok=True)

    emotion_cycle = cfg.emotion_cycle

    # Mapa canal
    ch_to_group = {}
    for gname, chs in cfg.channel_groups.items():
        for ch in chs:
            ch_to_group[ch] = gname

    for ch, evs_raw in per_channel_events.items():
        signal = df[ch].values
        th = thresholds[ch]
        group = ch_to_group.get(ch, "")

        fig, ax = plt.subplots(figsize=(20, 4))
        ax.plot(time, signal, color="gray", lw=0.6, alpha=0.9, label="Envelope RMS")
        ax.axhline(th, color="red", ls="--", lw=1.1, alpha=0.85,label=f"Umbral = {th:.2e}")

        # Sombras de epocas consenso
        for idx, ev in enumerate(consensus_events):
            emo = emotion_cycle[idx % len(emotion_cycle)]
            color = EMOTION_COLORS[emo]
            block = (idx // len(emotion_cycle)) + 1
            t0 = ev["onset_t"] - cfg.pre_margin_sec
            t1 = ev["offset_t"] + cfg.post_margin_sec
            ax.axvspan(t0, t1, color=color, alpha=0.12)
            ax.axvspan(ev["onset_t"], ev["offset_t"], color=color, alpha=0.30)
            ypos = signal.max() * 0.88
            ax.text(
                (ev["onset_t"] + ev["offset_t"]) / 2, ypos,
                f"{emo[:3]}\nB{block}", ha="center", va="center",
                fontsize=6.5, weight="bold", color=color,
            )

        # Marcadores de onset/offset crudos del canal
        for ev_raw in evs_raw:
            ax.axvline(ev_raw["onset_t"], color="navy", lw=0.7, alpha=0.55, ls=":")
            ax.axvline(ev_raw["offset_t"], color="darkorange", lw=0.7, alpha=0.55, ls=":")

        # Leyenda
        emo_patches = [mpatches.Patch(color=c, label=e) for e, c in EMOTION_COLORS.items()]
        raw_lines = [
            plt.Line2D([0], [0], color="navy", ls=":", lw=1, label="Onset canal"),
            plt.Line2D([0], [0], color="darkorange", ls=":", lw=1, label="Offset canal"),
        ]
        ax.legend(handles=emo_patches + raw_lines + ax.get_lines()[:2], loc="upper right", fontsize=7.5, ncol=2)

        ch_label = ch.replace("_Envelope_RMS", "")
        ax.set_title(f"{ch_label}  [{group}]  — Segmentacion por Gesticulacion", fontsize=11, weight="bold")
        ax.set_xlabel("Tiempo (s)", fontsize=9)
        ax.set_ylabel("Amplitud RMS", fontsize=9)
        ax.set_xlim(time[0], time[-1])
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        out = outhpath / f"verificacion_{ch_label}.png"
        fig.savefig(out, dpi=cfg.plot_dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> verificacion_{ch_label}.png")


def plot_comparative(df, time, thresholds, consensus_events, cfg):
    root = Path(cfg.output_path)
    outhpath = root/"Active_Channels"
    
    outhpath.mkdir(parents=True, exist_ok=True)

    # Ordenar canales
    ch_order = []
    for chs in cfg.channel_groups.values():
        ch_order.extend(chs)
    # Solo canales que existen en el df
    ch_order = [ch for ch in ch_order if ch in df.columns]

    # Mapa canal
    ch_to_group = {}
    for gname, chs in cfg.channel_groups.items():
        for ch in chs:
            ch_to_group[ch] = gname

    n_ch = len(ch_order)
    emotion_cycle = cfg.emotion_cycle

    fig, axes = plt.subplots(n_ch, 1, figsize=(22, 3.8 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    prev_group = None
    for ax, ch in zip(axes, ch_order):
        signal = df[ch].values
        th = thresholds[ch]
        group = ch_to_group.get(ch, "")
        gcolor = GROUP_COLORS.get(group, "gray")

        # Separador visual entre grupos 
        if group != prev_group and prev_group is not None:
            ax.spines["top"].set_linewidth(2.0)
            ax.spines["top"].set_color(gcolor)
        prev_group = group

        ax.plot(time, signal, color="gray", lw=0.55, alpha=0.92)
        ax.axhline(th, color="red", ls="--", lw=0.9, alpha=0.8, label=f"Umbral {th:.2e}")

        for idx, ev in enumerate(consensus_events):
            emo = emotion_cycle[idx % len(emotion_cycle)]
            color = EMOTION_COLORS[emo]
            ax.axvspan(ev["onset_t"]  - cfg.pre_margin_sec,
                       ev["offset_t"] + cfg.post_margin_sec,
                       color=color, alpha=0.10)
            ax.axvspan(ev["onset_t"], ev["offset_t"], color=color, alpha=0.28)

        ch_label = ch.replace("_Envelope_RMS", "")
        ax.set_ylabel(f"{ch_label}\n({group})", fontsize=8.5, weight="bold",color=gcolor)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.22)
        ax.set_xlim(time[0], time[-1])

        # Minileyenda de umbral en cada subplot
        ax.legend(loc="upper right", fontsize=7, framealpha=0.6)

    # Leyenda de emociones en el ultimo subplot
    emo_patches = [mpatches.Patch(color=c, label=e) for e, c in EMOTION_COLORS.items()]
    axes[-1].legend(handles=emo_patches, loc="lower right", fontsize=8.5, framealpha=0.85)
    axes[-1].set_xlabel("Tiempo (s)", fontsize=10)

    fig.suptitle("Segmentación EMG — 4 Canales (Comparativa por Grupo Muscular)", fontsize=13, weight="bold", y=1.005)
    fig.tight_layout()

    out = outhpath / "verificacion_comparativa_4canales.png"
    fig.savefig(out, dpi=cfg.plot_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> verificacion_comparativa_4canales.png")


def plot_group_summary(group_events, consensus_events, time, df, cfg):
    root = Path(cfg.output_path)
    outhpath = root/"Active_Channels"
    
    outhpath.mkdir(parents=True, exist_ok=True)

    emotion_cycle = cfg.emotion_cycle
    group_names   = list(cfg.channel_groups.keys())
    n_groups      = len(group_names)

    fig, axes = plt.subplots(n_groups + 1, 1, figsize=(22, 3.5 * (n_groups + 1)), sharex=True)

    # Subplots por grupo
    for ax, gname in zip(axes[:n_groups], group_names):
        gcolor = GROUP_COLORS.get(gname, "gray")
        # Senal media del grupo 
        chs_in_group = [c for c in cfg.channel_groups[gname] if c in df.columns]
        if chs_in_group:
            mean_sig = df[chs_in_group].mean(axis=1).values
            # Normalizar para visualizacion conjunta
            peak = mean_sig.max()
            if peak > 0:
                mean_sig = mean_sig / peak
            ax.fill_between(time, mean_sig, alpha=0.25, color=gcolor)
            ax.plot(time, mean_sig, color=gcolor, lw=0.6, alpha=0.7)

        for ev in group_events.get(gname, []):
            ax.axvspan(ev["onset_t"], ev["offset_t"], color=gcolor, alpha=0.45)
            ax.axvline(ev["onset_t"], color=gcolor, lw=1.0, alpha=0.8)

        ax.set_ylabel(gname, fontsize=9, weight="bold", color=gcolor)
        ax.set_yticks([])
        ax.grid(True, alpha=0.2)
        ax.set_xlim(time[0], time[-1])

    # Subplot consenso
    ax_c = axes[-1]
    for idx, ev in enumerate(consensus_events):
        emo   = emotion_cycle[idx % len(emotion_cycle)]
        color = EMOTION_COLORS[emo]
        block = (idx // len(emotion_cycle)) + 1
        ax_c.axvspan(ev["onset_t"], ev["offset_t"], color=color, alpha=0.55)
        ax_c.text(
            (ev["onset_t"] + ev["offset_t"]) / 2, 0.5,
            f"{emo[:3]}\nB{block}", ha="center", va="center",
            fontsize=7, weight="bold", color="white",
        )

    emo_patches = [mpatches.Patch(color=c, label=e) for e, c in EMOTION_COLORS.items()]
    ax_c.legend(handles=emo_patches, loc="upper right", fontsize=8)
    ax_c.set_ylabel("Consenso\nFinal", fontsize=9, weight="bold")
    ax_c.set_yticks([])
    ax_c.set_xlabel("Tiempo (s)", fontsize=10)
    ax_c.grid(True, alpha=0.2)
    ax_c.set_xlim(time[0], time[-1])

    fig.suptitle("Diagnostico de Deteccion — Eventos por Grupo Muscular", fontsize=12, weight="bold", y=1.002)
    fig.tight_layout()

    out = outhpath / "diagnostico_grupos.png"
    fig.savefig(out, dpi=cfg.plot_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> diagnostico_grupos.png")


# Pipeline principal
def epoch_slicing(cfg):
    if cfg is None:
        cfg = Config()

    # 1. Carga de datos
    df = pd.read_csv(cfg.csv_path)
    time = df["Time_Seconds"].values
    fs = 1.0 / np.mean(np.diff(time))
    print(f"      {len(df):,} muestras  |  fs ≈ {fs:.1f} Hz  |  {time[-1]:.1f}s total")

    # 2. Umbrales por canal
    all_channels = [ch for chs in cfg.channel_groups.values() for ch in chs]

    # Mapa canal: k_onset y k_offset_ratio efectivos
    ch_to_k: dict[str, float] = {}
    ch_to_koff: dict[str, float] = {}
    for gname, chs in cfg.channel_groups.items():
        k = cfg.k_baseline_per_group.get(gname, cfg.k_baseline)
        koff = cfg.k_offset_ratio_per_group.get(gname, cfg.k_offset_ratio)
        for ch in chs:
            ch_to_k[ch] = k
            ch_to_koff[ch] = koff

    using_per_group = bool(cfg.k_baseline_per_group)
    baseline_kind = "movil (mediana+MAD)" if cfg.use_rolling_baseline else "fijo (ventana inicial)"
    if using_per_group:
        k_summary = ", ".join(f"{g}→k_on={v}" for g, v in cfg.k_baseline_per_group.items())
        print(f"\nUmbrales por canal — baseline {baseline_kind}, k por grupo ({k_summary}), "
              f"hysteresis k_off_ratio={cfg.k_offset_ratio}")
    else:
        print(f"\nUmbrales por canal — baseline {baseline_kind}, k = {cfg.k_baseline}, "
              f"hysteresis k_off_ratio={cfg.k_offset_ratio}")

    # Arrays de baseline/umbral por canal (mismo largo que la señal)
    baseline_med: dict[str, np.ndarray] = {}
    baseline_sigma: dict[str, np.ndarray] = {}
    upper_th: dict[str, np.ndarray] = {}
    lower_th: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}   # valor representativo (mediana) solo para graficas/logs

    for ch in all_channels:
        k_eff = ch_to_k.get(ch, cfg.k_baseline)
        koff_eff = ch_to_koff.get(ch, cfg.k_offset_ratio)
        sig = df[ch].values

        if cfg.use_rolling_baseline:
            med, sigma = rolling_baseline_stats(
                sig, fs, cfg.rolling_baseline_window_sec, cfg.rolling_baseline_min_frac)
        else:
            th_fixed = compute_threshold(sig, time, cfg.baseline_window_sec, k_eff)
            mask = time <= cfg.baseline_window_sec
            med = np.full_like(sig, sig[mask].mean(), dtype=float)
            sigma = np.full_like(sig, max(sig[mask].std(), 1e-12), dtype=float)

        upper, lower = hysteresis_thresholds(med, sigma, k_eff, koff_eff)

        baseline_med[ch]   = med
        baseline_sigma[ch] = sigma
        upper_th[ch]       = upper
        lower_th[ch]       = lower
        thresholds[ch]     = float(np.median(upper))  # representativo para plots/logs

        print(f"      {ch} (k_on={k_eff}, k_off_ratio={koff_eff}): "
              f"umbral_on≈{thresholds[ch]:.4e} (mediana)")

    # 3. Deteccion por canal (umbral doble + filtros de forma)
    print(f"\nDetección de activaciones por canal "
          f"(ROI: {cfg.t_start}–{cfg.t_end}s)")
    per_channel_events: dict[str, list[dict]] = {}
    for ch in all_channels:
        evs = detect_channel_events(
            df[ch].values, time,
            upper_th[ch], lower_th[ch],
            baseline_med[ch], baseline_sigma[ch],
            fs, cfg)
        per_channel_events[ch] = evs
        print(f"      {ch}: {len(evs):2d} activaciones")

    # 4. Consenso por grupos musculares
    print(f"\nConsenso por grupos musculares "
          f"(≥ {cfg.min_groups_active} grupo(s) activo(s))")
    group_events    = detect_group_events(per_channel_events, cfg)
    consensus_events = build_consensus_events(group_events, cfg)

    for gname, gevs in group_events.items():
        print(f"      {gname}: {len(gevs):2d} eventos de grupo")

    total_expected = cfg.num_blocks * len(cfg.emotion_cycle)
    n_detected     = len(consensus_events)
    status = "✓" if n_detected == total_expected else "⚠"
    print(f"\n      {status} Eventos detectados (pasada global): {n_detected} / {total_expected} esperados")

    # 4b. Refinamiento con baseline local (5s previos al countdown de cada gesto)
    if cfg.use_local_baseline_refinement:
        print(f"\nRefinando onset/offset con baseline local "
              f"({cfg.local_baseline_sec}s previos al countdown de {cfg.countdown_sec}s)")
        consensus_events = refine_consensus_events(consensus_events, df, time, fs, cfg)

    # 5. Extraccion y exportacion 
    epochs = extract_epochs(df, time, consensus_events, cfg)
    export_csvs(epochs, cfg)
    df_metrics = export_metrics(consensus_events, cfg)
    print(f"      {len(epochs)} activaciones guardadas\n")
    print(df_metrics.to_string(index=False))

    # 6. Graficas
    plot_per_channel(df, time, per_channel_events, thresholds, consensus_events, cfg)
    plot_comparative(df, time, thresholds, consensus_events, cfg)
    plot_group_summary(group_events, consensus_events, time, df, cfg)

    return df_metrics


# Punto de entrada
if __name__ == "__main__":
    cfg = Config(
        csv_path = "../Test1/Data/FREEEMG_Processed_Signals.csv",
        output_path = "../Test1",

        # Protocolo
        num_blocks = 5,

        # Topología muscular

        channel_groups = {
            "Grupo_A": ["EMG1_Envelope_RMS", "EMG4_Envelope_RMS"],
            "Grupo_B":  ["EMG2_Envelope_RMS", "EMG3_Envelope_RMS"],
        },
        min_groups_active = 1,   # 1 = basta con que un grupo lo detecte

        # Umbral por grupo muscular (k de encendido)
        k_baseline_per_group = {
            "Grupo_A": 3.0,   
            "Grupo_B": 4.5,  
        },
        k_baseline = 10.0,

        # Baseline movil robusto (mediana + MAD), reemplaza el baseline fijo
        use_rolling_baseline = False,
        rolling_baseline_window_sec = 20.0,
        baseline_window_sec = 20.0,     # usado solo si use_rolling_baseline=False

        # Umbral doble / hysteresis: apagado mas laxo que encendido
        k_offset_ratio = 0.6,
        k_offset_ratio_per_group = {
            "Grupo_A": 0.6,
            "Grupo_B": 0.6,
        },

        # Refinamiento con baseline local pre-countdown (5s antes del countdown)
        use_local_baseline_refinement = False,
        countdown_sec = 3.0,
        local_baseline_sec = 5.0,
        local_refine_search_margin_sec = 2.0,

        # Filtros de forma: energia y factor de cresta
        use_shape_filters = False,
        energy_ratio_min = 2.0,
        crest_factor_min = 1.05,
        crest_factor_max = 5.0,

        # Rango1 temporal de la grabacion util
        t_start = 0.0,
        t_end = 700.0,
        min_inter_event_sec = 10.0,     # Separacion minima entre gestos (s)
        tolerance_sec = 3.0,            # Tolerancia para asociar canales entre si

        # Forma de onda y gap filling
        smoothing_window_sec = 0.5,     # Suavizado antes de detectar cruces
        gap_fill_sec = 1.5,             # Brecha maxima dentro de un mismo gesto
        min_event_dur_sec = 2.5,        # Minimo post-gap-filling 
        max_event_dur_sec = 10.0,       # Maximo (gestos no duran mas de esto)

        # Márgenes de la epoca exportada
        pre_margin_sec  = 1.0,
        post_margin_sec = 1.0,
    )
    epoch_slicing(cfg)