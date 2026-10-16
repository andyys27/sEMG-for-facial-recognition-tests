"""
Logica de deteccion de eventos:
  1. detect_channel_events    -> activaciones por canal individual (hysteresis + forma)
  2. detect_group_events      -> consenso dentro de cada grupo muscular
  3. build_consensus_events   -> consenso final entre grupos
  4. refine_consensus_events  -> ajuste fino de onset/offset con baseline local
                                 (reposo justo antes del countdown de cada gesto)
"""

from dataclasses import replace
import numpy as np
from .signal_utils import smooth_signal, event_shape_metrics, hysteresis_thresholds


# Helper: fusion/dedup por descanso real entre eventos (no por onset-onset)
def _merge_two_events(a, b):
    # Combina dos eventos que se solapan o estan demasiado pegados en uno solo
    base = a if a.get("peak_amp", 0) >= b.get("peak_amp", 0) else b
    merged = dict(base)
    merged["onset_t"] = min(a["onset_t"], b["onset_t"])
    merged["offset_t"] = max(a["offset_t"], b["offset_t"])
    if "peak_amp" in a and "peak_amp" in b:
        merged["peak_amp"] = max(a["peak_amp"], b["peak_amp"])
    for key in ("channels_active", "groups_active"):
        if key in a or key in b:
            vals = list(a.get(key, [])) + list(b.get(key, []))
            merged[key] = sorted(set(vals))
    if "groups_active" in merged:
        merged["n_groups"] = len(merged["groups_active"])
    return merged


def dedupe_by_gap(events, min_gap_sec):
    # Depura/fusiona eventos consecutivos usando la separacion REAL de reposo
    if not events:
        return events
    events = sorted(events, key=lambda e: e["onset_t"])
    clean = [dict(events[0])]
    for ev in events[1:]:
        prev = clean[-1]
        gap = ev["onset_t"] - prev["offset_t"]
        if gap < min_gap_sec:
            clean[-1] = _merge_two_events(prev, ev)
        else:
            clean.append(dict(ev))
    return clean


# 1. Deteccion por canal individual
def detect_channel_events(signal, time, upper, lower, baseline_med, baseline_sigma, fs, cfg, channel_name=None):
    tag = f"[{channel_name}] " if channel_name else ""
    sig_smooth = smooth_signal(signal, fs, cfg.smoothing_window_sec)
    roi = (time >= cfg.t_start) & (time <= cfg.t_end)   # Region de interes

    # 1. Deteccion con hysteresis: enciende al cruzar upper, apaga al cruzar lower
    raw_events = []
    in_event = False
    onset_idx = None

    for i in range(1, len(sig_smooth)):
        if not roi[i]:
            if in_event:
                in_event = False
                onset_idx = None
            continue

        rising = sig_smooth[i] > upper[i] and sig_smooth[i - 1] <= upper[i - 1]
        falling = sig_smooth[i] < lower[i] and sig_smooth[i - 1] >= lower[i - 1]

        if not in_event and rising:
            in_event = True
            onset_idx = i
        elif in_event and falling:
            raw_events.append((onset_idx, i - 1))
            in_event = False
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
        accept_duration = cfg.min_event_dur_sec <= dur <= cfg.max_event_dur_sec

        intense_override = False
        intense_ratio = None
        if not accept_duration and cfg.allow_short_intense_events and dur <= cfg.max_event_dur_sec:
            if cfg.min_event_dur_sec_intense <= dur < cfg.min_event_dur_sec:
                peak_local = float(signal[on:off + 1].max())
                intense_ratio = peak_local / max(upper[on], 1e-12)
                if intense_ratio >= cfg.intense_peak_ratio:
                    intense_override = True

        if not (accept_duration or intense_override):
            if cfg.debug_rejections:
                razon = "duracion_excede_max" if dur > cfg.max_event_dur_sec else "duracion_bajo_min"
                print(f"      {tag}descartado t=[{time[on]:.2f},{time[off]:.2f}]s "
                      f"dur={dur:.2f}s razon={razon} "
                      f"(limites: {cfg.min_event_dur_sec}-{cfg.max_event_dur_sec}s)")
            continue

        if intense_override and cfg.debug_rejections:
            print(f"      {tag}rescatado por intensidad t=[{time[on]:.2f},{time[off]:.2f}]s "
                  f"dur={dur:.2f}s pico/umbral={intense_ratio:.1f}x (min {cfg.intense_peak_ratio}x)")

        seg = signal[on:off + 1]

        if cfg.use_shape_filters:
            energy_ratio, crest_factor = event_shape_metrics(
                seg, baseline_sigma[on:off + 1], baseline_med[on:off + 1])
            ok_energy = energy_ratio >= cfg.energy_ratio_min
            ok_crest = cfg.crest_factor_min <= crest_factor <= cfg.crest_factor_max
            if not (ok_energy and ok_crest):
                if cfg.debug_rejections:
                    print(f"      {tag}descartado t=[{time[on]:.2f},{time[off]:.2f}]s "
                          f"dur={dur:.2f}s razon=forma "
                          f"energy_ratio={energy_ratio:.2f}(min {cfg.energy_ratio_min}) "
                          f"crest_factor={crest_factor:.2f}(rango {cfg.crest_factor_min}-{cfg.crest_factor_max})")
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

    # 4. Deduplicacion por descanso real (offset_anterior -> onset_siguiente)
    return dedupe_by_gap(events, cfg.min_inter_event_sec)


# 2. Consenso por grupos musculares
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

            g_onset = min(e["onset_t"] for _, e in cluster)
            g_offset = max(e["offset_t"] for _, e in cluster)
            g_peak = max(e["peak_amp"] for _, e in cluster)
            merged.append({
                "onset_t": g_onset,
                "offset_t": g_offset,
                "peak_amp": g_peak,
                "channels_active": [ch for ch, _ in cluster],
            })

        # Deduplicar por descanso real dentro del grupo
        dedup = dedupe_by_gap(merged, cfg.min_inter_event_sec)

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

        c_onset = min(e["onset_t"] for _, e in cluster)
        c_offset = max(e["offset_t"] for _, e in cluster)

        consensus.append({
            "onset_t": c_onset,
            "offset_t": c_offset,
            "groups_active": active_groups,
            "n_groups": len(active_groups),
            "channels_active": [
                ch for _, ev in cluster
                for ch in ev["channels_active"]
            ],
        })

    # Ordenar y depurar solapamientos/eventos pegados por descanso real
    if not consensus:
        return []

    return dedupe_by_gap(consensus, cfg.min_inter_event_sec)


# 3. Refinamiento con baseline local (5s previos al countdown de cada gesto)
def refine_event_local_baseline(signal, time, fs, onset_t, offset_t, cfg, k_on, k_off_ratio):
    # Redefine onset/offset de UN evento usando como baseline los `local_baseline_sec`
    # segundos de reposo justo antes del countdown que precede al gesto 
    
    reposo_end = onset_t - cfg.countdown_sec
    reposo_calc_end = reposo_end - cfg.local_refine_safety_margin_sec
    reposo_start = reposo_calc_end - cfg.local_baseline_sec
    mask_baseline = (time >= reposo_start) & (time < reposo_calc_end)

    if mask_baseline.sum() < max(5, int(fs)):
        return onset_t, offset_t, None, None

    baseline_seg = signal[mask_baseline]
    med_local = float(np.median(baseline_seg))
    mad_local = float(np.median(np.abs(baseline_seg - med_local))) * 1.4826
    mad_local = max(mad_local, 1e-12)

    upper = med_local + k_on * mad_local
    lower = med_local + (k_on * k_off_ratio) * mad_local

    # Buscar el cruce real en una ventana amplia alrededor del evento original
    search_start = reposo_end - cfg.local_refine_search_pre_buffer_sec
    search_end = offset_t + cfg.local_refine_search_margin_sec
    mask_search = (time >= search_start) & (time <= search_end)
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
    # Aplica el refinamiento local a cada evento de consenso, canal por canal
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
            new_on, new_off, med_local, mad_local = refine_event_local_baseline(
                df[ch].values, time, fs,
                ev["onset_t"], ev["offset_t"], cfg,
                ch_to_k[ch], ch_to_koff[ch])
            onsets.append(new_on)
            offsets.append(new_off)

            if cfg.debug_rejections:
                moved_on = abs(new_on - ev["onset_t"]) > 0.05
                moved_off = abs(new_off - ev["offset_t"]) > 0.05
                if med_local is None:
                    print(f"      [{ch}] refinamiento local: sin reposo suficiente "
                          f"antes de t={ev['onset_t']:.2f}s — se conserva original")
                elif moved_on or moved_off:
                    print(f"      [{ch}] refinado: onset {ev['onset_t']:.2f}->{new_on:.2f}s "
                          f"offset {ev['offset_t']:.2f}->{new_off:.2f}s "
                          f"(baseline_local={med_local:.3e}, MAD={mad_local:.3e})")

        ev_ref = dict(ev)
        ev_ref["onset_t"] = float(np.min(onsets))
        ev_ref["offset_t"] = float(np.max(offsets))
        if cfg.debug_rejections and (abs(ev_ref["onset_t"] - ev["onset_t"]) > 0.05
                                      or abs(ev_ref["offset_t"] - ev["offset_t"]) > 0.05):
            print(f"      >> Evento consenso refinado: "
                  f"[{ev['onset_t']:.2f},{ev['offset_t']:.2f}]s -> "
                  f"[{ev_ref['onset_t']:.2f},{ev_ref['offset_t']:.2f}]s")
        refined.append(ev_ref)

    refined.sort(key=lambda x: x["onset_t"])
    return refined


# 4. Rescate automatico dentro de huecos anomalos
def rescue_gaps(consensus_events, df, time, fs, cfg, baseline_med, baseline_sigma, ch_to_k, ch_to_koff, all_channels):
    # Recorre los huecos entre eventos consecutivos ya detectados con el umbral normal
    for ev in consensus_events:
        ev.setdefault("rescued_from_gap", False)

    if not cfg.gap_rescue_enabled or len(consensus_events) < 2:
        return consensus_events

    consensus_events = sorted(consensus_events, key=lambda e: e["onset_t"])
    rescued = []

    for prev, nxt in zip(consensus_events[:-1], consensus_events[1:]):
        gap = nxt["onset_t"] - prev["offset_t"]
        if gap <= cfg.expected_gap_max_sec:
            continue

        window_start = prev["offset_t"]
        window_end = nxt["onset_t"]
        cfg_window = replace(cfg, t_start=window_start, t_end=window_end)

        per_channel_window = {}
        for ch in all_channels:
            k_relaxed = ch_to_k[ch] * cfg.gap_rescue_k_factor
            upper_r, lower_r = hysteresis_thresholds(
                baseline_med[ch], baseline_sigma[ch], k_relaxed, ch_to_koff[ch])
            evs = detect_channel_events(
                df[ch].values, time, upper_r, lower_r,
                baseline_med[ch], baseline_sigma[ch], fs, cfg_window,
                channel_name=f"{ch}(rescate {window_start:.1f}-{window_end:.1f}s, k={k_relaxed:.2f})")
            per_channel_window[ch] = evs

        group_events_window = detect_group_events(per_channel_window, cfg_window)
        consensus_window = build_consensus_events(group_events_window, cfg_window)

        if consensus_window and cfg.debug_rejections:
            print(f"      >> Rescate en hueco [{window_start:.1f},{window_end:.1f}]s: "
                  f"{len(consensus_window)} evento(s) recuperado(s) con k_on relajado")

        for ev in consensus_window:
            ev["rescued_from_gap"] = True
            rescued.append(ev)

    if not rescued:
        return consensus_events

    combined = consensus_events + rescued
    combined.sort(key=lambda e: e["onset_t"])
    return combined