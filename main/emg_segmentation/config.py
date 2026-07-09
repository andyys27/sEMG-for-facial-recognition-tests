"""
Configuracion central del pipeline de segmentacion EMG.
Ajuste de una rutas, protocolo, umbrales, duraciones y margenes
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict

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
        "Grupo_B": ["EMG2_Envelope_RMS", "EMG3_Envelope_RMS"],
    })
 
    # Minimo de grupos que deben coincidir para validar un evento consenso
    min_groups_active: int = 1
 
    # Baseline y umbral
    baseline_window_sec: float = 20.0          # usado solo si use_rolling_baseline=False
    k_baseline: float = 3.0
    k_baseline_per_group: dict[str, float] = field(default_factory=dict)
 
    # Baseline movil robusto (mediana + MAD) 
    use_rolling_baseline: bool = True
    rolling_baseline_window_sec: float = 20.0   # ventana causal para mediana/MAD
    rolling_baseline_min_frac: float = 0.25     # min_periods como fraccion de la ventana
 
    # Umbral doble (hysteresis / Schmitt trigger)
    # umbral_offset = baseline + (k_onset * k_offset_ratio) * sigma  (k_offset_ratio < 1)
    k_offset_ratio: float = 0.6
    k_offset_ratio_per_group: dict[str, float] = field(default_factory=dict)
 
    # Refinamiento con baseline local pre-countdown 
    use_local_baseline_refinement: bool = True
    countdown_sec: float = 3.0          # duracion del countdown antes del gesto
    local_baseline_sec: float = 5.0     # ventana de reposo previa al countdown a usar
    local_refine_search_margin_sec: float = 2.0
 
    # Filtros de forma (energia y factor de cresta)
    use_shape_filters: bool = True
    energy_ratio_min: float = 3.0       # energia_evento / energia_ruido_baseline minima
    crest_factor_min: float = 1.05      # rechaza mesetas casi planas (probable ruido)
    crest_factor_max: float = 8.0       # rechaza picos puntiagudos (probable artefacto)
 
    # Rango de interes
    t_start: float = 0.0
    t_end: float = 380.0
 
    # Suavizado del envelope
    smoothing_window_sec: float = 0.5
 
    # Separacion minima entre eventos consecutivos
    min_inter_event_sec: float = 10.0
 
    # Tolerancia temporal para asociar eventos de diferentes grupos/canales
    tolerance_sec: float = 3.0
 
    # Gap filling
    gap_fill_sec: float = 1.5
 
    # Duracion valida de un evento por canal
    min_event_dur_sec: float = 2.5
    max_event_dur_sec: float = 10.0
 
    # Margenes al extraer la epoca
    pre_margin_sec: float = 1.0
    post_margin_sec: float = 1.0
 
    # DPI graficas
    plot_dpi: int = 200
 
    # Debug: loguear eventos candidatos rechazados 
    debug_rejections: bool = False
 
    # Etiquetado del CSV completo (processed + label) 
    export_labeled_full_csv: bool = True
    labeled_csv_name: str = "processed_labeled.csv"
    reposo_label: str = "Reposo"
    countdown_label_prefix: str = "Countdown_"

    save_config_json: bool = True

    def save(self, filepath: str):
        path = Path(filepath).resolve()
        
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4, ensure_ascii=False)
            
        print(f"\n[Configuración guardada en la ruta:\n -> {path}\n")