from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pipeline import emg_processing_pipeline

# Rutas de Test
base_path = Path("../Test1")

# Rutas derivadas
data_path = base_path / "Data" / "FREEEMG_EMG_with_timestamp.csv"
analysis_path = base_path / "Analysis"
output_path = base_path / "Data" / "FREEEMG_Processed_Signals.csv"

# Crear carpeta de analisis
analysis_path.mkdir(parents=True, exist_ok=True)

# Carga de datos
data = pd.read_csv(data_path)
data = data.fillna(0) 

# Convertir timestamp a ms
timestamps = pd.to_datetime(data.iloc[:, 0])
time = (timestamps - timestamps.iloc[0]).dt.total_seconds()

data.iloc[:, 1:] = data.iloc[:, 1:].fillna(0)

# Frecuencia de muestreo promedio
fs = 1.0 / np.mean(np.diff(time))
print(f"Frecuencia de muestreo: {fs:.2f} Hz")

# Cantidad de canales EMG
emg_channels = [c for c in data.columns if "EMG" in c]

# Diccionario para creacion de CSV
processed_data_dict = {
    "Timestamp": data.iloc[:, 0].values,
    "Time_Seconds": time
}

# Procesamiento por canal
for col in emg_channels:
    raw_signal = data[col].values
    sig = emg_processing_pipeline(raw_signal, fs)

    # Diccionario de analisis
    processed_data_dict[f"{col}_Raw"] = sig["raw"]
    processed_data_dict[f"{col}_Filtered"] = sig["filtered"]
    processed_data_dict[f"{col}_Envelope_RMS"] = sig["envelope"]
    processed_data_dict[f"{col}_Normalized_Z"] = sig["normalized"]
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"Análisis Completo de Canal: {col}", fontsize=14, fontweight='bold')
    
    # Raw vs Fourier Raw
    axes[0, 0].plot(time, sig["raw"], color='tab:blue', label="Original")
    axes[0, 0].set_title("Señal Original (Raw)")
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    axes[0, 1].plot(sig["raw xf"], sig["raw mag"], color='tab:red', label="Espectro")
    axes[0, 1].set_title("Análisis de Fourier (Raw)")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # Filtered vs Fourier Clean
    axes[1, 0].plot(time, sig["filtered"], color='tab:orange', label="Filtrada")
    axes[1, 0].set_title("Señal Filtrada (Bandpass + Notch)")
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    axes[1, 1].plot(sig["clean xf"], sig["clean mag"], color='tab:green', label="Espectro Limpio")
    axes[1, 1].set_title("Análisis de Fourier (Limpia)")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    # Envolvente vs Normalizada
    axes[2, 0].plot(time, sig["envelope"], color='tab:purple', label="Envolvente")
    axes[2, 0].set_title("Envolvente RMS")
    axes[2, 0].grid(True)
    axes[2, 0].legend()

    axes[2, 1].plot(time, sig["normalized"], color='tab:gray', label="Z-score")
    axes[2, 1].set_title("Envolvente Normalizada")
    axes[2, 1].grid(True)
    axes[2, 1].legend()
    
    plt.tight_layout()
    plt.savefig(analysis_path/f"Análisis_{col}.png")
    plt.close()

processed_df = pd.DataFrame(processed_data_dict)
processed_df.to_csv(output_path, index=False)