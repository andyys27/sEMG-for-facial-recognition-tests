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
        csv_path = str(root / "Test1/Data/FREEEMG_Processed_Signals.csv"),
        output_path = str(root / "Test1"),

        # Protocolo
        num_blocks=5,

        # Topologia muscular
        channel_groups={
            "Grupo_A": ["EMG1_Envelope_RMS", "EMG4_Envelope_RMS"],
            "Grupo_B": ["EMG2_Envelope_RMS", "EMG3_Envelope_RMS"],
        },
        min_groups_active=1,   # 1 = basta con que un grupo lo detecte

        # Umbral por grupo muscular (k de encendido)
        k_baseline_per_group={
            "Grupo_A": 3.0,
            "Grupo_B": 4.5,
        },
        k_baseline=10.0,

        # Baseline movil robusto (mediana + MAD), reemplaza el baseline fijo
        use_rolling_baseline=False,
        rolling_baseline_window_sec=20.0,
        baseline_window_sec=20.0,     # usado solo si use_rolling_baseline=False

        # Umbral doble / hysteresis: apagado mas laxo que encendido
        k_offset_ratio=0.6,
        k_offset_ratio_per_group={
            "Grupo_A": 0.6,
            "Grupo_B": 0.6,
        },

        # Refinamiento con baseline local pre-countdown (5s antes del countdown)
        use_local_baseline_refinement=False,
        countdown_sec=3.0,
        local_baseline_sec=5.0,
        local_refine_search_margin_sec=2.0,

        # Filtros de forma: energia y factor de cresta
        use_shape_filters=False,
        energy_ratio_min=3.0,
        crest_factor_min=1.05,
        crest_factor_max=8.0,

        # Rango temporal de la grabacion util
        t_start=0.0,
        t_end=700.0,
        min_inter_event_sec=10.0,     # Separacion minima entre gestos (s)
        tolerance_sec=3.0,            # Tolerancia para asociar canales entre si

        # Forma de onda y gap filling
        smoothing_window_sec=0.5,     # Suavizado antes de detectar cruces
        gap_fill_sec=1.5,             # Brecha maxima dentro de un mismo gesto
        min_event_dur_sec=2.5,        # Minimo post-gap-filling
        max_event_dur_sec=10.0,       # Maximo (gestos no duran mas de esto)

        # Margenes de la epoca exportada
        pre_margin_sec=1.0,
        post_margin_sec=1.0,

        # CSV completo etiquetado (Reposo / Countdown_Emocion / <Emocion>)
        export_labeled_full_csv=True,
        labeled_csv_name="Data/processed_labeled.csv",
    )
    epoch_slicing(cfg)