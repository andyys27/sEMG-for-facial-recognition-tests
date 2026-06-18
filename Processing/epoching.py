import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def epoch_slicing(csv_path, output_path=".", num_blocks=10, k_baseline=2.5):
    # Carga de datos
    df = pd.read_csv(csv_path)
    time = df["Time_Seconds"].values

    # Frecuencia de muestreo
    fs = 1.0 / np.mean(np.diff(time))
    
    # Aproximacion de inicio y fin de medicion
    T_START = 35.0
    T_END = 755.0
    
    # Canales de EMG
    emg_channels = ["EMG1_Envelope_RMS", "EMG2_Envelope_RMS", "EMG3_Envelope_RMS", "EMG4_Envelope_RMS"]
    
    # Promediacion de canales EMG
    master_signal = df[emg_channels].mean(axis=1).values
    
    # Senal Suavizada con media móvil para deteccion de picos
    master_smooth = pd.Series(master_signal).rolling(window=int(1.0 * fs), center=True).mean().fillna(0).values
    
    # Calculo de umbral de reposo utilizando tiempo inicial
    idx_rest = time <= 20.0
    rest_mean = master_signal[idx_rest].mean()
    rest_std = master_signal[idx_rest].std()
    threshold = rest_mean + (k_baseline * rest_std)
        
    # Deteccion de bloques de activacion muscular
    df_useful = df[(df["Time_Seconds"] >= T_START) & (df["Time_Seconds"] <= T_END)]
    indices_peaks_useful, properties = find_peaks(
        master_smooth[df_useful.index], 
        height=threshold, 
        # Distancia de 12 segundos para no duplicar 
        distance=int(12.0 * fs),        
        prominence=master_smooth.max() * 0.05
    )
    
    # Mapeo de los indices locales de vuelta a los indices globales
    indices_peaks = df_useful.index[indices_peaks_useful].tolist()
    total_expected = num_blocks * 4
    
    # Control de deteccion de picos de mayor amplitud y prolongacion
    if len(indices_peaks) > total_expected:
        prominences = properties.get('prominences', np.ones(len(indices_peaks)))
        top_indices = np.argsort(prominences)[-total_expected:]
        indices_peaks = sorted([indices_peaks[i] for i in top_indices])
        
    print(f"Activaciones musculares detectadas: {len(indices_peaks)} / {total_expected}")
    
    # Algoritmo de ventana para asignacion de etiquetas
    emotion_cycle = ["Sonrisa", "Disgusto", "Sorprendido", "Triste"]
    master_list = []
    
    for idx, idx_peak in enumerate(indices_peaks):
        emotion_name = emotion_cycle[idx % 4]
        current_block = (idx // 4) + 1
        
        # Busqueda hacia atras hasta cruzar el umbral 
        start_idx = idx_peak
        left_limit = max(0, idx_peak - int(4.5 * fs))
        for i in range(idx_peak, left_limit, -1):
            if master_signal[i] <= threshold:
                start_idx = i
                break
            start_idx = i 
            
        # Busqueda hacia adelante buscando el regreso al reposo 
        end_idx = idx_peak
        right_limit = min(len(df) - 1, idx_peak + int(6.5 * fs))
        for i in range(idx_peak, right_limit):
            if master_signal[i] <= threshold and i > idx_peak + int(1.5 * fs):
                end_idx = i
                break
            end_idx = i
            
        # Margenes minimos para deteccion completa de activacion
        start_final = max(0, start_idx - int(0.5 * fs))     
        end_final = min(len(df) - 1, end_idx + int(1.2 * fs)) 
        
        # Extracción del bloque, clonacion y calculo del tiempo relativo de la época
        sub_df = df.iloc[start_final:end_final+1].copy()
        sub_df["Block"] = int(current_block)
        sub_df["Emotion"] = emotion_name
        sub_df["Epoch_Time"] = sub_df["Time_Seconds"] - time[start_final]
        
        # Exportacion individual del bloque directo 
        sub_df.to_csv(f"{output_path}/{emotion_name}/bloque_{current_block}.csv", index=False)
        master_list.append(sub_df)
        
    # Resultados por emocion
    if len(master_list) == total_expected:
        df_total = pd.concat(master_list, ignore_index=True)
        for emo in emotion_cycle:
            df_total[df_total["Emotion"] == emo].to_csv(f"{output_path}/{emo}/{emo}_Total.csv", index=False)
        
    # Grafica de verificacion
    plt.figure(figsize=(16, 6))
    plt.plot(time, master_signal, color="darkgray", alpha=0.5, label="Señal EMG")
    plt.axhline(y=threshold, color="red", linestyle="--", alpha=0.7, label="Umbral de eposo")
    
    for idx, sub_df in enumerate(master_list):
        t_i = sub_df["Time_Seconds"].min()
        t_f = sub_df["Time_Seconds"].max()
        emo = sub_df["Emotion"].iloc[0]
        bloque = sub_df["Block"].iloc[0]
        
        plt.axvspan(t_i, t_f, color="mediumseagreen", alpha=0.2, edgecolor="green", linewidth=1.2)
        plt.text((t_i + t_f) / 2, master_signal.max() * 0.7, 
                 f"{emo}\nB{bloque}", ha='center', va='center', fontsize=8, weight='bold', color="darkgreen")
        
    plt.title("Segmentación por emociones")
    plt.xlabel("Tiempo (seg)")
    plt.ylabel("Amplitud promedio")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.savefig(f"{output_path}/Control_epoch.png", dpi=300, bbox_inches='tight')
    plt.show()

epoch_slicing(
    csv_path="../Test1/Data/FREEEMG_Processed_Signals.csv",
    output_path="../Test1",
    num_blocks=10,
    k_baseline=2.5
)