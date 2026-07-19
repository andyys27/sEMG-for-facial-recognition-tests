"""
Punto de entrada: Se define la configuracion de la corrida y se invoca 
la funcion principal de slicing de epocas

python -m main.run_segmentation
"""

from pathlib import Path
from main.emg_segmentation import epoch_slicing
from main.emg_segmentation import Config

# Raiz del proyecto
root = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    cfg = Config(
        csv_path = str(root / "Test2/Data/FREEEMG_Processed_Signals.csv"),
        output_path = str(root / "Test2"),

        # Protocolo
        num_blocks=3,

        # Topologia muscular
        channel_groups={
            "Grupo_A": ["EMG1_Envelope_RMS", "EMG2_Envelope_RMS"],
            "Grupo_B": ["EMG3_Envelope_RMS", "EMG4_Envelope_RMS"],
        },
        min_groups_active=1,   # 1 = basta con que un grupo lo detecte

        # Umbral por grupo muscular (k de encendido)
        k_baseline_per_group={
            "Grupo_A": 4.5,
            "Grupo_B": 5.0,
        },
        k_baseline=10.0,

        # Baseline movil robusto (mediana + MAD), reemplaza el baseline fijo
        use_rolling_baseline=False,
        rolling_baseline_window_sec=20.0,   # ventana causal para mediana/MAD
        baseline_window_sec=20.0,           # usado solo si use_rolling_baseline=False

        # Umbral doble / hysteresis: apagado mas laxo que encendido
        k_offset_ratio=0.6,
        k_offset_ratio_per_group={
            "Grupo_A": 0.6,
            "Grupo_B": 0.2,
        },

        # Refinamiento con baseline local pre-countdown (5s antes del countdown)
        use_local_baseline_refinement=True,
        countdown_sec=3.0,                      # duracion del countdown antes del gesto
        local_baseline_sec=8.0,                 # ventana de reposo previa al countdown a usar
        local_refine_search_margin_sec=5.0,

        # Separa donde se muestrea el reposo de donde se empieza a buscar el cruce real
        local_refine_safety_margin_sec=1.5,
        local_refine_search_pre_buffer_sec=6.0,

        # Filtros de forma: energia y factor de cresta
        use_shape_filters=False,
        energy_ratio_min=3.0,       # energia_evento / energia_ruido_baseline minima
        crest_factor_min=1.05,      # rechaza mesetas casi planas (probable ruido)
        crest_factor_max=8.0,       # rechaza picos puntiagudos (probable artefacto)

        # Rango temporal de la grabacion util
        t_start=0.0,
        t_end=330.0,
        min_inter_event_sec=10.0,       # Separacion minima entre gestos (s)
        tolerance_sec=3.0,              # Tolerancia para asociar canales entre si

        # Forma de onda y gap filling
        smoothing_window_sec=0.5,       # Suavizado antes de detectar cruces

        # Gap filling
        gap_fill_sec=3.0,               # Brecha maxima dentro de un mismo gesto

        # Duracion valida de un evento por canal
        min_event_dur_sec=2.0,          # Minimo post-gap-filling
        max_event_dur_sec=22.0,         # Maximo (gestos no duran mas de esto)

        # Excepcion: eventos cortos pero muy intensos 
        allow_short_intense_events= True,
        min_event_dur_sec_intense=1.0,      # piso absoluto, por debajo de esto nunca se acepta
        intense_peak_ratio=2.0,             # pico debe superar el umbral_on por este factor

        # Margenes de la epoca exportada
        pre_margin_sec=1.0,
        post_margin_sec=1.0,

        # Sanity check: marcar huecos entre eventos fuera de lo esperado 
        expected_gap_min_sec=3.0,
        expected_gap_max_sec=25.0,

        # Rescate automatico dentro de huecos anomalos
        gap_rescue_enabled=True,
        gap_rescue_k_factor=0.9,

        # CSV completo etiquetado (Reposo / Countdown_Emocion / <Emocion>)
        export_labeled_full_csv=True,
        labeled_csv_name="Data/processed_labeled.csv",

        debug_rejections=True,
        save_config_json=False,
    )
    epoch_slicing(cfg)
    if cfg.save_config_json:
        cfg.save("Test2/config_test2.json")