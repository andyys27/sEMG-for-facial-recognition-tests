"""Gráficas y tablas de diagnóstico de señal. Usa la API orientada a objetos de matplotlib
(Figure), sin pyplot, para no depender de ni alterar el backend global."""
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy.ndimage import uniform_filter1d

ORDER = ["Reposo", "Countdown", "Sonrisa", "Disgusto", "Sorprendido", "Triste"]
COLORS = {"Reposo": "#d9d9d9", "Countdown": "#ffe699", "Sonrisa": "#6baed6",
          "Disgusto": "#74c476", "Sorprendido": "#fd8d3c", "Triste": "#b49ad6",
          "Fuera_protocolo": "#ffffff"}


def envelope_matrix(x: np.ndarray, fs_hz: float, win_ms: float = 100.0) -> np.ndarray:
    """Envolvente RMS móvil de cada canal: (n_samples, n_channels)."""
    k = max(1, int(round(win_ms * fs_hz / 1000)))
    return np.sqrt(uniform_filter1d(np.asarray(x, dtype="float64") ** 2, size=k, axis=0, mode="nearest"))


def _activation(filtered: pd.DataFrame, channels: list[str], fs_hz: float) -> np.ndarray:
    """Envolvente de cada canal dividida por su mediana (1 ≈ nivel de reposo)."""
    env = envelope_matrix(filtered[channels].to_numpy(), fs_hz)
    return env / np.median(env, axis=0)


def plot_overview(filtered, intervals, seg_table, channels, fs_hz, out_path: Path) -> None:
    """Activación por canal en todo el protocolo, con fases sombreadas y segmentos usados."""
    t = filtered["t_ns"].to_numpy()
    t0, t1 = intervals["t_start_ns"].min(), intervals["t_end_ns"].max()
    m = (t >= t0 - 2e9) & (t <= t1 + 2e9)
    step = max(1, int(fs_hz / 50))
    ts = (t[m][::step] - t0) / 1e9
    act = _activation(filtered, channels, fs_hz)[m][::step]

    n = len(channels)
    fig = Figure(figsize=(16, 2.0 * n + 1.8), layout="constrained")
    axes = fig.subplots(n + 1, 1, sharex=True, gridspec_kw={"height_ratios": [3] * n + [0.7]})
    for j, ch in enumerate(channels):
        ax = axes[j]
        for _, r in intervals.iterrows():
            ax.axvspan((r.t_start_ns - t0) / 1e9, (r.t_end_ns - t0) / 1e9,
                       color=COLORS[r.label], alpha=0.6, lw=0)
        ax.semilogy(ts, act[:, j], lw=0.7, color="black")
        ax.set_ylabel(f"{ch}\n(× mediana)")
        ax.grid(alpha=0.3, which="both", lw=0.3)
    ax = axes[-1]
    for _, r in seg_table.iterrows():
        ax.barh(0, (r.t_end_ns - r.t_start_ns) / 1e9, left=(r.t_start_ns - t0) / 1e9, height=0.8,
                color=COLORS[r.label], edgecolor="black" if r.valid else "red", lw=0.8)
    ax.set_yticks([]); ax.set_ylabel("segmentos\nusados"); ax.set_xlabel("tiempo desde el inicio del protocolo (s)")
    fig.legend(handles=[Patch(color=COLORS[l], label=l) for l in ORDER], loc="outside upper center", ncol=6)
    fig.savefig(out_path, dpi=100)


def activation_profile(filtered, events, channels, fs_hz, edges=None) -> pd.DataFrame:
    """Activación (canal más activo / su mediana) promediada por trial, alineada a
    `hold_start` (inicio del gesto) y a `rest_start` posterior al gesto. Por gesto y total."""
    edges = np.arange(-1.0, 5.01, 0.25) if edges is None else edges
    act = _activation(filtered, channels, fs_hz).max(axis=1)
    t = filtered["t_ns"].to_numpy()
    anchors = {"hold_start": events[events.event == "hold_start"],
               "rest_start": events[(events.event == "rest_start") & (events.reason == "after_gesture")]}
    rows = []
    for anchor, ev in anchors.items():
        groups = {"todos": ev, **{g: d for g, d in ev.groupby("gesture")}}
        for g, d in groups.items():
            for a, b in zip(edges[:-1], edges[1:]):
                vals = []
                for t_ev in d["t_ns"]:
                    i0, i1 = np.searchsorted(t, [t_ev + a * 1e9, t_ev + b * 1e9])
                    if i1 > i0:
                        vals.append(act[i0:i1].mean())
                rows.append((anchor, g, a, b, float(np.mean(vals)) if vals else np.nan, len(vals)))
    return pd.DataFrame(rows, columns=["anchor", "gesture", "t_from", "t_to", "activation", "n_trials"])


def plot_activation_profile(profile: pd.DataFrame, margins: dict, params: dict, out_path: Path) -> None:
    """Perfil de activación por gesto con la zona que conservan los márgenes (verde)."""
    hold, rest = params.get("hold_s", 3.0), params.get("rest_s", 5.0)
    g_m, r_m = margins.get("Gesto", {}), margins.get("Reposo", {})
    spans = {"hold_start": (g_m.get("onset_s", 0), hold - g_m.get("offset_s", 0)),
             "rest_start": (r_m.get("onset_s", 0), rest - r_m.get("offset_s", 0))}
    titles = {"hold_start": "Alineado a hold_start (inicio del gesto)",
              "rest_start": "Alineado a rest_start (fin del gesto)"}
    fig = Figure(figsize=(14, 4.8), layout="constrained")
    for ax, anchor in zip(fig.subplots(1, 2, sharey=True), ["hold_start", "rest_start"]):
        for g, d in profile[profile.anchor == anchor].groupby("gesture"):
            x = (d.t_from + d.t_to) / 2
            ax.semilogy(x, d.activation, lw=2.6 if g == "todos" else 1.4,
                        color="black" if g == "todos" else COLORS.get(g), label=g)
        lo, hi = spans[anchor]
        ax.axvspan(lo, hi, color="green", alpha=0.12, label="zona conservada")
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_title(titles[anchor]); ax.set_xlabel("segundos"); ax.grid(alpha=0.3, which="both", lw=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.axes[0].set_ylabel("activación (× nivel de reposo)")
    fig.savefig(out_path, dpi=110)


def activity_by_label(filtered, labels, channels, fs_hz) -> pd.DataFrame:
    """Envolvente media por etiqueta y canal, en veces la mediana del canal."""
    env = _activation(filtered, channels, fs_hz)
    lab = labels["label"].to_numpy()
    rows = {l: env[lab == l].mean(axis=0) for l in [*ORDER, "Fuera_protocolo"] if (lab == l).any()}
    return pd.DataFrame(rows, index=channels).T


def plot_activity_by_label(tab: pd.DataFrame, out_path: Path) -> None:
    fig = Figure(figsize=(6.5, 4.6), layout="constrained")
    ax = fig.subplots()
    im = ax.imshow(np.log10(tab.to_numpy()), cmap="viridis", aspect="auto")
    ax.set_xticks(range(tab.shape[1]), tab.columns); ax.set_yticks(range(tab.shape[0]), tab.index)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            ax.text(j, i, f"{tab.iat[i, j]:.1f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="log10(× mediana)")
    ax.set_title("Activación media por etiqueta y canal")
    fig.savefig(out_path, dpi=110)