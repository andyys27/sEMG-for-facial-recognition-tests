"""
Features de dominio del tiempo por ventana, calculadas sobre la senal filtrada

Referencia estandar en clasificacion de gestos EMG: Hudgins et al. (1993),
Phinyomark et al. (2012) — MAV, RMS, WL, ZC, SSC
"""

import numpy as np

def _zero_crossings(x, threshold):
    # Cuenta cruces por cero, ignorando fluctuaciones menores al umbral
    signs = np.sign(x)
    signs[np.abs(x) < threshold] = 0
    nonzero = signs[signs != 0]
    if len(nonzero) < 2:
        return 0
    return int(np.sum(np.diff(nonzero) != 0))


def _slope_sign_changes(x, threshold):
    # Cuenta cuantas veces la pendiente cambia de signo
    if len(x) < 3:
        return 0
    d = np.diff(x)
    changes = 0
    for i in range(1, len(d)):
        if d[i] * d[i - 1] < 0 and (abs(d[i]) > threshold or abs(d[i - 1]) > threshold):
            changes += 1
    return changes


def time_domain_features(x, noise_threshold=None):
    # Set de features de dominio del tiempo 
    x = np.asarray(x, dtype=float)
    if noise_threshold is None:
        # Umbral adaptativo: fraccion pequeña del rango de la propia ventana,
        # para no necesitar un valor absoluto fijo entre canales/sujetos
        noise_threshold = 0.05 * (np.max(np.abs(x)) + 1e-12)

    mav = float(np.mean(np.abs(x)))
    rms = float(np.sqrt(np.mean(x ** 2)))
    wl = float(np.sum(np.abs(np.diff(x))))
    var = float(np.var(x))
    zc = _zero_crossings(x, noise_threshold)
    ssc = _slope_sign_changes(x, noise_threshold)
    iemg = float(np.sum(np.abs(x)))

    return {
        "MAV": mav,
        "RMS": rms,
        "WL": wl,
        "VAR": var,
        "ZC": zc,
        "SSC": ssc,
        "IEMG": iemg,
    }


def compute_window_features(window_df, channel_groups):
    # Calcula todas las features de una ventana para los 4 canales +
    # features cruzadas entre grupos musculares  
    feats = {}
    envelope_means = {}
 
    for gname, chs in channel_groups.items():
        for idx, ch in enumerate(chs):
            base = ch.replace("_Envelope_RMS", "")
            role = f"{gname}_ch{idx + 1}"  # nombre estable entre sujetos
 
            filtered_col = f"{base}_Filtered"
            envelope_col = f"{base}_Envelope_RMS"
            zscore_col = f"{base}_Normalized_Z"
 
            if filtered_col in window_df.columns:
                td = time_domain_features(window_df[filtered_col].values)
                for k, v in td.items():
                    feats[f"{role}_{k}"] = v
 
            if envelope_col in window_df.columns:
                env = window_df[envelope_col].values
                feats[f"{role}_env_mean"] = float(np.mean(env))
                feats[f"{role}_env_max"] = float(np.max(env))
                feats[f"{role}_env_std"] = float(np.std(env))
                envelope_means[role] = float(np.mean(env))
 
            if zscore_col in window_df.columns:
                z = window_df[zscore_col].values
                feats[f"{role}_z_mean"] = float(np.mean(z))
                feats[f"{role}_z_max"] = float(np.max(z))
 
    # Features cruzadas entre grupos musculares capturan el patron de reclutamiento
    # diferencial entre grupos que ya usamos para distinguir emociones
    group_names = list(channel_groups.keys())
    group_means = {}
    for gname, chs in channel_groups.items():
        roles = [f"{gname}_ch{i + 1}" for i in range(len(chs))]
        vals = [envelope_means[r] for r in roles if r in envelope_means]
        if vals:
            group_means[gname] = float(np.mean(vals))
 
    if len(group_names) == 2 and all(g in group_means for g in group_names):
        gA, gB = group_names
        denom = group_means[gB] if group_means[gB] > 1e-12 else 1e-12
        feats[f"ratio_{gA}_{gB}"] = group_means[gA] / denom
        feats[f"diff_{gA}_{gB}"] = group_means[gA] - group_means[gB]
 
    return feats