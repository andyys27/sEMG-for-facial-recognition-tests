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
    csv_path:    str = "../Test2/Data/FREEEMG_Processed_Signals.csv"
    output_path: str = "../Test2"

    # Protocolo
    num_blocks: int = 5
    emotion_cycle: list[str] = field(default_factory=lambda: [
        "Sonrisa", "Disgusto", "Sorprendido", "Triste"
    ])

    # Grupos musculares: cada lista agrupa canales equivalentes anatómicamente.
    # Un evento de grupo se activa si AL MENOS UNO de sus canales supera umbral.
    channel_groups: dict[str, list[str]] = field(default_factory=lambda: {
        "Grupo_Lateral": ["EMG1_Envelope_RMS", "EMG2_Envelope_RMS"],
        "Grupo_Medial":  ["EMG3_Envelope_RMS", "EMG4_Envelope_RMS"],
    })

# Mínimo de grupos que deben coincidir para validar un evento consenso
    min_groups_active: int = 1

    # Baseline y umbral
    baseline_window_sec:  float = 20.0
    k_baseline:           float = 3.0
    k_baseline_per_group: dict[str, float] = field(default_factory=dict)

    # Rango de interés (excluye reposo inicial/final de grabación)
    t_start: float = 0.0
    t_end:   float = 380.0

    # Suavizado del envelope antes de detectar cruces de umbral
    smoothing_window_sec: float = 0.5

    # Separación mínima entre eventos consecutivos
    min_inter_event_sec: float = 10.0

    # Tolerancia temporal para asociar eventos de diferentes grupos/canales
    tolerance_sec: float = 3.0

    # Gap filling
    gap_fill_sec: float = 1.5

    # Duración válida de un evento por canal (DESPUÉS del gap filling)
    min_event_dur_sec: float = 2.5
    max_event_dur_sec: float = 10.0

    # Márgenes al extraer la época
    pre_margin_sec:  float = 0.4
    post_margin_sec: float = 0.8

    # DPI gráficas
    plot_dpi: int = 200


# Utilidades de señal
def compute_threshold(signal: np.ndarray, time: np.ndarray,
                      baseline_window_sec: float, k: float) -> float:
    """Umbral adaptativo por canal: mean + k·std sobre ventana de reposo inicial."""
    mask = time <= baseline_window_sec
    if mask.sum() < 10:
        raise ValueError(
            f"Muy pocas muestras en ventana de baseline "
            f"({mask.sum()} muestras, t ≤ {baseline_window_sec}s). "
            "Verifica que baseline_window_sec sea menor al inicio del primer gesto."
        )
    baseline = signal[mask]
    return float(baseline.mean() + k * baseline.std())


def smooth_signal(signal: np.ndarray, fs: float, window_sec: float) -> np.ndarray:
    """Media móvil centrada. Rellena bordes con ffill/bfill."""
    w = max(1, int(window_sec * fs))
    return pd.Series(signal).rolling(window=w, center=True).mean().ffill().bfill().values


# Detección por canal individual
def detect_channel_events(
    signal: np.ndarray,
    time:   np.ndarray,
    threshold: float,
    fs:     float,
    cfg:    Config,
) -> list[dict]:
    sig_smooth = smooth_signal(signal, fs, cfg.smoothing_window_sec)
    active     = sig_smooth > threshold
    roi        = (time >= cfg.t_start) & (time <= cfg.t_end)

    # 1. Detectar todos los segmentos activos sin filtro de duración
    raw_events = []
    in_event   = False
    onset_idx  = None

    for i in range(1, len(active)):
        if not roi[i]:
            if in_event:
                in_event  = False
                onset_idx = None
            continue

        rising  = active[i] and not active[i - 1]
        falling = (not active[i]) and active[i - 1]

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
            merged[-1][1] = off          # extender el evento actual
        else:
            merged.append([on, off])

    # 3. Filtro de duración y cálculo de amplitudes 
    events = []
    for (on, off) in merged:
        dur = time[off] - time[on]
        if cfg.min_event_dur_sec <= dur <= cfg.max_event_dur_sec:
            seg = signal[on:off + 1]
            events.append({
                "onset_idx":  on,
                "offset_idx": off,
                "onset_t":    float(time[on]),
                "offset_t":   float(time[off]),
                "peak_amp":   float(seg.max()),
                "mean_amp":   float(seg.mean()),
            })

    if len(events) <= 1:
        return events

    # 4. Deduplicación por distancia mínima
    filtered = [events[0]]
    for ev in events[1:]:
        prev = filtered[-1]
        if ev["onset_t"] - prev["onset_t"] >= cfg.min_inter_event_sec:
            filtered.append(ev)
        elif ev["peak_amp"] > prev["peak_amp"]:
            filtered[-1] = ev

    return filtered


# Consenso por grupos musculares
def detect_group_events(
    per_channel_events: dict[str, list[dict]],
    cfg: Config,
) -> dict[str, list[dict]]:
    group_events: dict[str, list[dict]] = {}

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
        used   = set()

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

        # Deduplicar por distancia mínima dentro del grupo
        merged.sort(key=lambda x: x["onset_t"])
        dedup = [merged[0]]
        for ev in merged[1:]:
            if ev["onset_t"] - dedup[-1]["onset_t"] >= cfg.min_inter_event_sec:
                dedup.append(ev)
            elif ev["peak_amp"] > dedup[-1]["peak_amp"]:
                dedup[-1] = ev

        group_events[group_name] = dedup

    return group_events


def build_consensus_events(
    group_events: dict[str, list[dict]],
    cfg: Config,
) -> list[dict]:
    group_names = list(group_events.keys())

    # Pool de todos los eventos de todos los grupos
    pool = []
    for gname in group_names:
        for ev in group_events[gname]:
            pool.append((ev["onset_t"], gname, ev))
    pool.sort(key=lambda x: x[0])

    used      = set()
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


# Extracción de épocas
def extract_epochs(
    df: pd.DataFrame,
    time: np.ndarray,
    consensus_events: list[dict],
    cfg: Config,
) -> list[pd.DataFrame]:
    emotion_cycle = cfg.emotion_cycle
    epochs = []

    for idx, ev in enumerate(consensus_events):
        emotion = emotion_cycle[idx % len(emotion_cycle)]
        block   = (idx // len(emotion_cycle)) + 1

        t_start = ev["onset_t"]  - cfg.pre_margin_sec
        t_end   = ev["offset_t"] + cfg.post_margin_sec
        mask    = (time >= t_start) & (time <= t_end)

        if mask.sum() == 0:
            print(f"  ⚠ Evento {idx+1} ({emotion} B{block}): sin muestras "
                  f"en [{t_start:.2f}, {t_end:.2f}]s — omitido.")
            continue

        sub = df[mask].copy()
        sub["Block"]           = block
        sub["Emotion"]         = emotion
        sub["Epoch_Time"]      = sub["Time_Seconds"].values - time[np.where(mask)[0][0]]
        sub["N_Groups"]        = ev["n_groups"]
        sub["Active_Channels"] = ", ".join(sorted(set(ev["channels_active"])))
        epochs.append(sub)

    return epochs


# Exportación
def export_csvs(epochs: list[pd.DataFrame], cfg: Config) -> None:
    """CSV individual por bloque/emoción + total por emoción."""
    root = Path(cfg.output_path)
    for emo in cfg.emotion_cycle:
        (root / emo).mkdir(parents=True, exist_ok=True)

    for sub in epochs:
        emo   = sub["Emotion"].iloc[0]
        block = int(sub["Block"].iloc[0])
        sub.to_csv(root / emo / f"bloque_{block:02d}.csv", index=False)

    if epochs:
        df_all = pd.concat(epochs, ignore_index=True)
        for emo in cfg.emotion_cycle:
            df_emo = df_all[df_all["Emotion"] == emo]
            if not df_emo.empty:
                df_emo.to_csv(root / emo / f"{emo}_Total.csv", index=False)


def export_metrics(
    consensus_events: list[dict],
    thresholds:       dict[str, float],
    cfg: Config,
) -> pd.DataFrame:
    """Reporte de detección exportado solo como CSV."""
    root          = Path(cfg.output_path)
    emotion_cycle = cfg.emotion_cycle
    rows = []

    for idx, ev in enumerate(consensus_events):
        emo   = emotion_cycle[idx % len(emotion_cycle)]
        block = (idx // len(emotion_cycle)) + 1
        rows.append({
            "Event_Index":     idx + 1,
            "Block":           block,
            "Emotion":         emo,
            "Onset_s":         round(ev["onset_t"], 4),
            "Offset_s":        round(ev["offset_t"], 4),
            "Duration_s":      round(ev["offset_t"] - ev["onset_t"], 4),
            "N_Groups":        ev["n_groups"],
            "Groups_Active":   ", ".join(sorted(ev["groups_active"])),
            "Channels_Active": ", ".join(sorted(set(ev["channels_active"]))),
        })

    # Añadir fila de thresholds al pie como metadatos legibles
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(root / "detection_metrics.csv", index=False)

    print(f"\n  → detection_metrics.csv   ({len(rows)} eventos)")
    return df_metrics


# Gráficas
EMOTION_COLORS = {
    "Sonrisa":     "#27ae60",
    "Disgusto":    "#e74c3c",
    "Sorprendido": "#2980b9",
    "Triste":      "#8e44ad",
}

# Colores por grupo muscular para gráficas individuales
GROUP_COLORS = {
    "Grupo_A": "#e67e22",
    "Grupo_B":  "#16a085",
}


def plot_per_channel(
    df: pd.DataFrame,
    time: np.ndarray,
    per_channel_events: dict[str, list[dict]],
    thresholds: dict[str, float],
    consensus_events: list[dict],
    cfg: Config,
) -> None:
    root          = Path(cfg.output_path)
    emotion_cycle = cfg.emotion_cycle

    # Mapa canal → grupo para título informativo
    ch_to_group = {}
    for gname, chs in cfg.channel_groups.items():
        for ch in chs:
            ch_to_group[ch] = gname

    for ch, evs_raw in per_channel_events.items():
        signal = df[ch].values
        th     = thresholds[ch]
        group  = ch_to_group.get(ch, "")

        fig, ax = plt.subplots(figsize=(20, 4))
        ax.plot(time, signal, color="#333333", lw=0.6, alpha=0.9, label="Envelope RMS")
        ax.axhline(th, color="#c0392b", ls="--", lw=1.1, alpha=0.85,
                   label=f"Umbral = {th:.2e}")

        # Sombras de épocas consenso
        for idx, ev in enumerate(consensus_events):
            emo   = emotion_cycle[idx % len(emotion_cycle)]
            color = EMOTION_COLORS[emo]
            block = (idx // len(emotion_cycle)) + 1
            t0    = ev["onset_t"]  - cfg.pre_margin_sec
            t1    = ev["offset_t"] + cfg.post_margin_sec
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
            ax.axvline(ev_raw["onset_t"],  color="navy",       lw=0.7, alpha=0.55, ls=":")
            ax.axvline(ev_raw["offset_t"], color="darkorange", lw=0.7, alpha=0.55, ls=":")

        # Leyenda
        emo_patches = [mpatches.Patch(color=c, label=e)
                       for e, c in EMOTION_COLORS.items()]
        raw_lines = [
            plt.Line2D([0], [0], color="navy",       ls=":", lw=1, label="Onset canal"),
            plt.Line2D([0], [0], color="darkorange", ls=":", lw=1, label="Offset canal"),
        ]
        ax.legend(handles=emo_patches + raw_lines + ax.get_lines()[:2],
                  loc="upper right", fontsize=7.5, ncol=2)

        ch_label = ch.replace("_Envelope_RMS", "")
        ax.set_title(f"{ch_label}  [{group}]  — Segmentación por Emociones",
                     fontsize=11, weight="bold")
        ax.set_xlabel("Tiempo (s)", fontsize=9)
        ax.set_ylabel("Amplitud RMS", fontsize=9)
        ax.set_xlim(time[0], time[-1])
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        out = root / f"verificacion_{ch_label}.png"
        fig.savefig(out, dpi=cfg.plot_dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  → verificacion_{ch_label}.png")


def plot_comparative(
    df: pd.DataFrame,
    time: np.ndarray,
    thresholds: dict[str, float],
    consensus_events: list[dict],
    cfg: Config,
) -> None:
    root   = Path(cfg.output_path)
    # Ordenar canales: primero grupo lateral, luego medial
    ch_order = []
    for chs in cfg.channel_groups.values():
        ch_order.extend(chs)
    # Solo canales que existen en el df
    ch_order = [ch for ch in ch_order if ch in df.columns]

    # Mapa canal → grupo
    ch_to_group = {}
    for gname, chs in cfg.channel_groups.items():
        for ch in chs:
            ch_to_group[ch] = gname

    n_ch  = len(ch_order)
    emotion_cycle = cfg.emotion_cycle

    fig, axes = plt.subplots(n_ch, 1, figsize=(22, 3.8 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    prev_group = None
    for ax, ch in zip(axes, ch_order):
        signal = df[ch].values
        th     = thresholds[ch]
        group  = ch_to_group.get(ch, "")
        gcolor = GROUP_COLORS.get(group, "#555555")

        # Separador visual entre grupos (línea horizontal en el eje y=0)
        if group != prev_group and prev_group is not None:
            ax.spines["top"].set_linewidth(2.0)
            ax.spines["top"].set_color(gcolor)
        prev_group = group

        ax.plot(time, signal, color="#3d3d3d", lw=0.55, alpha=0.92)
        ax.axhline(th, color="#c0392b", ls="--", lw=0.9, alpha=0.8,
                   label=f"Umbral {th:.2e}")

        for idx, ev in enumerate(consensus_events):
            emo   = emotion_cycle[idx % len(emotion_cycle)]
            color = EMOTION_COLORS[emo]
            ax.axvspan(ev["onset_t"]  - cfg.pre_margin_sec,
                       ev["offset_t"] + cfg.post_margin_sec,
                       color=color, alpha=0.10)
            ax.axvspan(ev["onset_t"], ev["offset_t"], color=color, alpha=0.28)

        ch_label = ch.replace("_Envelope_RMS", "")
        ax.set_ylabel(f"{ch_label}\n({group})", fontsize=8.5, weight="bold",
                      color=gcolor)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.22)
        ax.set_xlim(time[0], time[-1])

        # Mini-leyenda de umbral en cada subplot
        ax.legend(loc="upper right", fontsize=7, framealpha=0.6)

    # Leyenda de emociones en el último subplot
    emo_patches = [mpatches.Patch(color=c, label=e)
                   for e, c in EMOTION_COLORS.items()]
    axes[-1].legend(handles=emo_patches, loc="lower right",
                    fontsize=8.5, framealpha=0.85)
    axes[-1].set_xlabel("Tiempo (s)", fontsize=10)

    fig.suptitle("Segmentación EMG — 4 Canales (Comparativa por Grupo Muscular)",
                 fontsize=13, weight="bold", y=1.005)
    fig.tight_layout()

    out = root / "verificacion_comparativa_4canales.png"
    fig.savefig(out, dpi=cfg.plot_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → verificacion_comparativa_4canales.png")


def plot_group_summary(
    group_events:     dict[str, list[dict]],
    consensus_events: list[dict],
    time:             np.ndarray,
    df:               pd.DataFrame,
    cfg:              Config,
) -> None:
    root          = Path(cfg.output_path)
    emotion_cycle = cfg.emotion_cycle
    group_names   = list(cfg.channel_groups.keys())
    n_groups      = len(group_names)

    fig, axes = plt.subplots(n_groups + 1, 1,
                              figsize=(22, 3.5 * (n_groups + 1)), sharex=True)

    # Subplots por grupo
    for ax, gname in zip(axes[:n_groups], group_names):
        gcolor = GROUP_COLORS.get(gname, "#555555")
        # Señal media del grupo 
        chs_in_group = [c for c in cfg.channel_groups[gname] if c in df.columns]
        if chs_in_group:
            mean_sig = df[chs_in_group].mean(axis=1).values
            # Normalizar para visualización conjunta
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

    fig.suptitle("Diagnóstico de Detección — Eventos por Grupo Muscular → Consenso",
                 fontsize=12, weight="bold", y=1.002)
    fig.tight_layout()

    out = root / "diagnostico_grupos.png"
    fig.savefig(out, dpi=cfg.plot_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → diagnostico_grupos.png")


# Pipeline principal
def epoch_slicing(cfg: Optional[Config] = None) -> pd.DataFrame:
    if cfg is None:
        cfg = Config()

    SEP = "=" * 62

    # 1. Carga
    df   = pd.read_csv(cfg.csv_path)
    time = df["Time_Seconds"].values
    fs   = 1.0 / np.mean(np.diff(time))
    print(f"      {len(df):,} muestras  |  fs ≈ {fs:.1f} Hz  |  {time[-1]:.1f}s total")

    # 2. Umbrales por canal
    all_channels = [ch for chs in cfg.channel_groups.values() for ch in chs]

    # Mapa canal → k efectivo
    ch_to_k: dict[str, float] = {}
    for gname, chs in cfg.channel_groups.items():
        k = cfg.k_baseline_per_group.get(gname, cfg.k_baseline)
        for ch in chs:
            ch_to_k[ch] = k

    using_per_group = bool(cfg.k_baseline_per_group)
    if using_per_group:
        k_summary = ", ".join(f"{g}→k={v}" for g, v in cfg.k_baseline_per_group.items())
        print(f"\nUmbrales por canal con k por grupo ({k_summary})")
    else:
        print(f"\nUmbrales por canal (baseline t ≤ {cfg.baseline_window_sec}s, k = {cfg.k_baseline})")

    thresholds: dict[str, float] = {}
    for ch in all_channels:
        k_eff = ch_to_k.get(ch, cfg.k_baseline)
        th = compute_threshold(df[ch].values, time, cfg.baseline_window_sec, k_eff)
        thresholds[ch] = th
        print(f"      {ch} (k={k_eff}): {th:.4e}")

    # 3. Detección por canal 
    print(f"\nDetección de activaciones por canal "
          f"(ROI: {cfg.t_start}–{cfg.t_end}s)")
    per_channel_events: dict[str, list[dict]] = {}
    for ch in thresholds:
        evs = detect_channel_events(df[ch].values, time, thresholds[ch], fs, cfg)
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
    print(f"\n      {status} Eventos detectados: {n_detected} / {total_expected} esperados")

    # 5. Extracción y exportación 
    epochs     = extract_epochs(df, time, consensus_events, cfg)
    export_csvs(epochs, cfg)
    df_metrics = export_metrics(consensus_events, thresholds, cfg)
    print(f"      {len(epochs)} épocas guardadas\n")
    print(df_metrics.to_string(index=False))

    # 6. Gráficas
    plot_per_channel(df, time, per_channel_events, thresholds, consensus_events, cfg)
    plot_comparative(df, time, thresholds, consensus_events, cfg)
    plot_group_summary(group_events, consensus_events, time, df, cfg)

    return df_metrics


# Punto de entrada
if __name__ == "__main__":
    cfg = Config(
        csv_path    = "../Test2/Data/FREEEMG_Processed_Signals.csv",
        output_path = "../Test2",

        # Protocolo
        num_blocks  = 10,

        # Topología muscular

        channel_groups = {
            "Grupo_A": ["EMG1_Envelope_RMS", "EMG2_Envelope_RMS"],
            "Grupo_B":  ["EMG3_Envelope_RMS", "EMG4_Envelope_RMS"],
        },
        min_groups_active = 1,   # 1 = basta con que un grupo lo detecte

        # Umbral por grupo muscular
        k_baseline_per_group = {
            "Grupo_A": 3.0,   
            "Grupo_B":  4.5,  
        },
        k_baseline          = 3.0,
        baseline_window_sec = 20.0,

        # Rango1 temporal de la grabación útil
        t_start             = 35.0,
        t_end               = 700.0,
        min_inter_event_sec = 10.0,   # Separación mínima entre gestos (s)
        tolerance_sec       = 3.0,    # Tolerancia para asociar canales entre sí

        # Forma de onda y gap filling
        smoothing_window_sec = 0.5,   # Suavizado antes de detectar cruces
        gap_fill_sec         = 1.5,   # Brecha máxima dentro de un mismo gesto
        min_event_dur_sec    = 2.5,   # Mínimo post-gap-filling (protocolo ≥ 3s)
        max_event_dur_sec    = 10.0,  # Máximo (gestos no duran más de esto)

        # Márgenes de la época exportada
        pre_margin_sec  = 0.4,
        post_margin_sec = 0.8,
    )
    epoch_slicing(cfg)