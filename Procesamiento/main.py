import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pipeline import emg_processing_pipeline

# Carga de datos
data = pd.read_csv("../Test1/Data/FREEEMG_EMG_with_timestamp.csv")
data = data.fillna(0) 

# Convertir timestamp a ms
timestamps = pd.to_datetime(data.iloc[:, 0])
time = (timestamps - timestamps.iloc[0]).dt.total_seconds()

# Extraer canales EMG
EMG1 = data.iloc[:, 1].values
EMG2 = data.iloc[:, 2].values
EMG3 = data.iloc[:, 3].values
EMG4 = data.iloc[:, 4].values

# Calculo de frecuencia de muestreo 
fs = 1.0 / np.mean(np.diff(time))

# Procesamiento de la señal EMG 
processed_EMG1 = emg_processing_pipeline(EMG1, fs)
processed_EMG2 = emg_processing_pipeline(EMG2, fs)
processed_EMG3 = emg_processing_pipeline(EMG3, fs)
processed_EMG4 = emg_processing_pipeline(EMG4, fs)

signals = [processed_EMG1, processed_EMG2, processed_EMG3, processed_EMG4]

# Visualizacion de resultados
for i, sig in enumerate(signals, 1):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(3,1,1)
    plt.plot(time, sig["raw"])
    plt.title(f"EMG{i} - Raw")
    plt.legend(loc="upper right")
    
    plt.subplot(3,1,2)
    plt.plot(time, sig["filtered"])
    plt.title(f"EMG{i} - Filtered")
    plt.legend(loc="upper right")
    
    plt.subplot(3,1,3)
    plt.plot(time, sig["envelope"])
    plt.title(f"EMG{i} - Envelope")
    plt.legend(loc="upper right")
    
    plt.tight_layout()
    plt.show()