"""
Pipeline principal: carga datos, calcula baseline/umbral por canal,
detecta activaciones, arma el consenso, refina onset/offset, exporta
epocas + CSV completo etiquetado, y genera las graficas de diagnostico.
"""

import numpy as np
import pandas as pd

from .config import Config
from .signal_utils import compute_threshold, rolling_baseline_stats, hysteresis_thresholds
from .detection import (
    detect_channel_events, detect_group_events,
    build_consensus_events, refine_consensus_events, rescue_gaps,
)
from .epochs import extract_epochs, export_csvs, export_metrics
from .labeling import build_labeled_dataframe, export_labeled_csv
from .plotting import plot_per_channel, plot_comparative, plot_group_summary


def epoch_slicing(cfg: Config = None):
    if cfg is None:
        cfg = Config()

    # 1. Carga de datos
    df = pd.read_csv(cfg.csv_path)
    time = df["Time_Seconds"].values
    fs = 1.0 / np.mean(np.diff(time))
    print(f"      {len(df):,} muestras  |  fs ≈ {fs:.1f} Hz  |  {time[-1]:.1f}s total")

    # 2. Umbrales por canal (baseline movil o fijo + hysteresis)
    all_channels = [ch for chs in cfg.channel_groups.values() for ch in chs]

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
            compute_threshold(sig, time, cfg.baseline_window_sec, k_eff)  # solo por compatibilidad/logs
            mask = time <= cfg.baseline_window_sec
            med = np.full_like(sig, sig[mask].mean(), dtype=float)
            sigma = np.full_like(sig, max(sig[mask].std(), 1e-12), dtype=float)

        upper, lower = hysteresis_thresholds(med, sigma, k_eff, koff_eff)

        baseline_med[ch] = med
        baseline_sigma[ch] = sigma
        upper_th[ch] = upper
        lower_th[ch] = lower
        thresholds[ch] = float(np.median(upper))  # representativo para plots/logs

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
            fs, cfg, channel_name=ch)
        per_channel_events[ch] = evs
        print(f"      {ch}: {len(evs):2d} activaciones")

    # 4. Consenso por grupos musculares
    print(f"\nConsenso por grupos musculares "
          f"(≥ {cfg.min_groups_active} grupo(s) activo(s))")
    group_events = detect_group_events(per_channel_events, cfg)
    consensus_events = build_consensus_events(group_events, cfg)

    for gname, gevs in group_events.items():
        print(f"      {gname}: {len(gevs):2d} eventos de grupo")

    total_expected = cfg.num_blocks * len(cfg.emotion_cycle)
    n_detected = len(consensus_events)
    status = "✓" if n_detected == total_expected else "⚠"
    print(f"\n      {status} Eventos detectados (pasada global): {n_detected} / {total_expected} esperados")

    # 4b. Refinamiento con baseline local (5s previos al countdown de cada gesto)
    if cfg.use_local_baseline_refinement:
        print(f"\nRefinando onset/offset con baseline local "
              f"({cfg.local_baseline_sec}s previos al countdown de {cfg.countdown_sec}s)")
        consensus_events = refine_consensus_events(consensus_events, df, time, fs, cfg)

    # 4c. Rescate automatico dentro de huecos anomalos (umbral relajado,
    #     restringido SOLO a la ventana del hueco — nunca toca el resto)
    if cfg.gap_rescue_enabled:
        n_before = len(consensus_events)
        consensus_events = rescue_gaps(
            consensus_events, df, time, fs, cfg,
            baseline_med, baseline_sigma, ch_to_k, ch_to_koff, all_channels)
        n_rescued = len(consensus_events) - n_before
        if n_rescued > 0:
            print(f"\n  >> {n_rescued} evento(s) recuperado(s) por rescate de huecos "
                  f"(k_on x{cfg.gap_rescue_k_factor})")

    total_expected = cfg.num_blocks * len(cfg.emotion_cycle)
    n_final = len(consensus_events)
    status = "✓" if n_final == total_expected else "⚠"
    print(f"      {status} Eventos finales (tras refinamiento y rescate): "
          f"{n_final} / {total_expected} esperados")

    # 5. Extraccion y exportacion de epocas individuales
    epochs = extract_epochs(df, time, consensus_events, cfg)
    export_csvs(epochs, cfg)
    df_metrics = export_metrics(consensus_events, cfg)
    print(f"      {len(epochs)} activaciones guardadas\n")
    print(df_metrics.to_string(index=False))

    # 5b. CSV completo del processed con label por muestra (Reposo/Countdown/Emocion)
    if cfg.export_labeled_full_csv:
        df_labeled = build_labeled_dataframe(df, time, consensus_events, cfg)
        export_labeled_csv(df_labeled, cfg)

    # 6. Graficas
    plot_per_channel(df, time, per_channel_events, thresholds, consensus_events, cfg)
    plot_comparative(df, time, thresholds, consensus_events, cfg)
    plot_group_summary(group_events, consensus_events, time, df, cfg)

    return df_metrics