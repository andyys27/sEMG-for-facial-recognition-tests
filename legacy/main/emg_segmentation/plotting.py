"""
Graficas de verificacion/diagnostico:
  - plot_per_channel:  una figura por canal, con umbral, eventos crudos y
                        epocas de consenso superpuestas.
  - plot_comparative:  los 4 canales apilados, agrupados visualmente por
                        grupo muscular.
  - plot_group_summary: senal promedio por grupo + linea de tiempo de
                        consenso final (bloque/emocion).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

EMOTION_COLORS = {
    "Sonrisa": "green",
    "Disgusto": "red",
    "Sorprendido": "blue",
    "Triste": "purple",
}

# Colores por grupo muscular para graficas individuales
GROUP_COLORS = {
    "Grupo_A": "orange",
    "Grupo_B": "teal",
}


def plot_per_channel(df, time, per_channel_events, thresholds, consensus_events, cfg):
    root = Path(cfg.output_path)
    outhpath = root / "Active_Channels"
    outhpath.mkdir(parents=True, exist_ok=True)

    emotion_cycle = cfg.emotion_cycle

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
        ax.axhline(th, color="red", ls="--", lw=1.1, alpha=0.85, label=f"Umbral = {th:.2e}")

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

        for ev_raw in evs_raw:
            ax.axvline(ev_raw["onset_t"], color="navy", lw=0.7, alpha=0.55, ls=":")
            ax.axvline(ev_raw["offset_t"], color="darkorange", lw=0.7, alpha=0.55, ls=":")

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
    outhpath = root / "Active_Channels"
    outhpath.mkdir(parents=True, exist_ok=True)

    ch_order = []
    for chs in cfg.channel_groups.values():
        ch_order.extend(chs)
    ch_order = [ch for ch in ch_order if ch in df.columns]

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

        if group != prev_group and prev_group is not None:
            ax.spines["top"].set_linewidth(2.0)
            ax.spines["top"].set_color(gcolor)
        prev_group = group

        ax.plot(time, signal, color="gray", lw=0.55, alpha=0.92)
        ax.axhline(th, color="red", ls="--", lw=0.9, alpha=0.8, label=f"Umbral {th:.2e}")

        for idx, ev in enumerate(consensus_events):
            emo = emotion_cycle[idx % len(emotion_cycle)]
            color = EMOTION_COLORS[emo]
            ax.axvspan(ev["onset_t"] - cfg.pre_margin_sec,
                       ev["offset_t"] + cfg.post_margin_sec,
                       color=color, alpha=0.10)
            ax.axvspan(ev["onset_t"], ev["offset_t"], color=color, alpha=0.28)

        ch_label = ch.replace("_Envelope_RMS", "")
        ax.set_ylabel(f"{ch_label}\n({group})", fontsize=8.5, weight="bold", color=gcolor)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.22)
        ax.set_xlim(time[0], time[-1])
        ax.legend(loc="upper right", fontsize=7, framealpha=0.6)

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
    outhpath = root / "Active_Channels"
    outhpath.mkdir(parents=True, exist_ok=True)

    emotion_cycle = cfg.emotion_cycle
    group_names = list(cfg.channel_groups.keys())
    n_groups = len(group_names)

    fig, axes = plt.subplots(n_groups + 1, 1, figsize=(22, 3.5 * (n_groups + 1)), sharex=True)

    for ax, gname in zip(axes[:n_groups], group_names):
        gcolor = GROUP_COLORS.get(gname, "gray")
        chs_in_group = [c for c in cfg.channel_groups[gname] if c in df.columns]
        if chs_in_group:
            mean_sig = df[chs_in_group].mean(axis=1).values
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

    ax_c = axes[-1]
    for idx, ev in enumerate(consensus_events):
        emo = emotion_cycle[idx % len(emotion_cycle)]
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