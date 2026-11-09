# Guía de notebooks para el proyecto sEMG

## Objetivo general

Los notebooks deben servir como una capa de análisis, diagnóstico y explicación del pipeline principal, no como un lugar donde hagamos lógica experimental sin trazabilidad. Su función es:

1. explicar y validar lo que ya está implementado en el pipeline;
2. mostrar visualmente el estado de los datos en `data/analysis/`;
3. documentar hipótesis sobre sincronización, señales, protocolos y generalización;
4. facilitar análisis extra que permitan interpretar los resultados sin reabrir el código fuente cada vez.

En otras palabras, un notebook debe ser una vista operativa del proyecto: reproducible, legible y conectada a las funciones reales del repositorio.

---

## Principios para todos los notebooks

Cada notebook debe cumplir estas reglas:

- Usar siempre la misma convención de nombres y rutas.
- Importar funciones desde `src/` y no duplicar lógica que ya existe.
- Hacer explícito el objetivo del notebook al inicio.
- Definir claramente qué sesión, sujeto o conjunto de datos se va a analizar.
- Mostrar validaciones rápidas de calidad antes de la interpretación.
- Guardar o mostrar resultados reproducibles en `data/analysis/` o en tablas/figuras del notebook.
- Terminarlos con conclusiones y con una lista de próximos pasos.

---

## Estructura recomendada de carpetas

```text
notebooks/
├── README.md                       # índice general de notebooks y lineamientos
├── eda/
│   ├── 01_exploracion_sesion.ipynb
│   ├── 02_sesion_completa_vs_fragmento.ipynb
│   ├── 03_imu_baiobit.ipynb
│   └── 04_analisis_por_canal.ipynb
├── label_validation/
│   ├── 01_validar_etiquetas.ipynb
│   ├── 02_validacion_por_trial.ipynb
│   ├── 03_comparacion_eventos_unity.ipynb
│   └── 04_summary_etiquetas.ipynb
├── synchronization/
│   ├── 01_revisar_sincronizacion.ipynb
│   ├── 02_drift_de_tiempo.ipynb
│   └── 03_validacion_imu_emg.ipynb
├── frequency/
│   ├── 01_fft_y_psd.ipynb
│   ├── 02_bandpower_por_gesto.ipynb
│   └── 03_rms_vs_frecuencia.ipynb
├── dataset/
│   ├── 01_quality_report.ipynb
│   ├── 02_distribucion_clases.ipynb
│   └── 03_segmentos_analisis.ipynb
├── model_comparison/
│   ├── 01_comparar_modelos.ipynb
│   ├── 02_confusion_analysis.ipynb
│   ├── 03_generalization_by_subject.ipynb
│   └── 04_error_analysis.ipynb
├── reporting/
│   ├── 01_summary_session_report.ipynb
│   ├── 02_summary_dataset.ipynb
│   └── 03_exploratory_report.ipynb
└── extra/
    ├── 01_analisis_experimental.ipynb
    └── 02_hypothesis_testing.ipynb
```

### Qué significa esta estructura

- `eda/`: exploración visual de la señal cruda, filtrada, metadata y sincronización.
- `label_validation/`: comprobación de que las etiquetas de Unity tienen sentido frente a la señal.
- `synchronization/`: confirmación de que los tiempos de EMG, IMU, Unity y eventos están alineados.
- `frequency/`: análisis de potencia, bandas y comportamiento espectral por gesto o sesión.
- `dataset/`: diagnósticos de calidad, clases, duración, balance y distribución segmentaria.
- `model_comparison/`: comparación de modelos, confusiones y estabilidad por sujeto.
- `reporting/`: notebooks con propósito narrativo para explicar los resultados en `data/analysis/`.
- `extra/`: análisis que no forman parte del pipeline principal, pero que ayudan a formular hipótesis y explicar resultados.

---

## Plantilla de contenido que debe tener cada notebook

Cada notebook debe seguir esta estructura mínima:

### 1. Título y objetivo

Debe incluir:

- nombre del notebook;
- qué pregunta responde;
- qué tipo de datos usa;
- qué resultado espera generar.

Ejemplo:

```markdown
## Objetivo
Analizar si la sincronización entre EMG y eventos de Unity es correcta para una sesión completa y si el protocolo genera etiquetas coherentes con la señal.
```

### 2. Contexto y alcance

Debemos responder:

- ¿Qué sesión o sujeto se está revisando?
- ¿Qué dato entra al notebook?
- ¿Qué dato sale del notebook?
- ¿Es un notebook de diagnóstico, visualización o comparación?

### 3. Carga de bibliotecas y configuración

Debe incluir:

- imports estándar de Python;
- imports del proyecto desde `src/`;
- variables globales como `SESSION_DIR`, `SUBJECT`, `CHANNELS`, `PLOT_STYLE`;
- opción para reproducibilidad.

### 4. Carga de sesión o datos

Se debe usar la API del proyecto, por ejemplo:

- `from src.io import load_session`;
- `from src.io.sync_timestamps import sync_timestamps`;
- `from src.protocol import label_session, protocol_params`;
- `from src.analysis.session_report import session_report`;
- `from src.analysis.signal_diagnostics import plot_signal_summary`.

No debe escribirse lógica duplicada para leer CSVs o reconstruir rutas si ya existe en `src/`.

### 5. Validaciones mínimas de calidad

Todo notebook debe validar antes de graficar:

- presencia de columnas esenciales;
- monotonicidad temporal;
- nulos/NaNs;
- tamaño de registros;
- coincidencia entre eventos y labels;
- si la sesión incluye IMU/Baiobit, revisar que los datos existan y estén sincronizados;
- si la señal se usa en frecuencia, comprobar que la secuencia tiene duración suficiente para FFT.

### 6. Análisis principal

Aquí va la parte de investigación. Debe ser clara y aislada. Ejemplos:

- EMG completo + eventos de Unity;
- comparación crudo vs filtrado;
- PSD por canal;
- activación por gesto vs reposo;
- validación visual por trial;
- proporción de clases por sujeto;
- confusión por modelo.

### 7. Interpretación y hallazgos

Debe incluir comentarios tipo:

- “La señal presenta un desplazamiento de fase en ...”;
- “La sincronización es consistente con Unity, pero hay ruido en ...”;
- “La clase X tiene menor separación entre sujetos.”

La interpretación debe ser una conclusión de investigación, no solo un conjunto de gráficas.

### 8. Salidas y artefactos

Cada notebook debe indicar:

- qué archivos genera;
- dónde los guarda;
- para qué sirven en `data/analysis/`.

Ejemplo:

```text
Output:
- data/analysis/session_report/session_01_quality.png
- data/analysis/reports/session_01_summary.md
- data/analysis/frequency/psd_by_channel.png
```

### 9. Conclusiones y próximos pasos

Debe cerrar con:

- resumen breve de lo que se encontró;
- si algo queda abierto;
- qué análisis adicional haría falta;
- qué notebook sigue la lógica recomendada.

---

## Dónde recoger funciones

La regla principal es: no reinventar funcionalidad.

### 1. Carga y sincronización

Usar:

- `src/io/load_emg.py`
- `src/io/load_events.py`
- `src/io/sync_timestamps.py`
- `src/io/interim.py`
- `src/io/_time.py`

Estas funciones deben proveer:

- carga de EMG;
- carga de eventos y protocol labels;
- sincronización entre tiempos diferenciales y absolutos;
- validación de relojes/timestamps.

### 2. Protocolo y etiquetas

Usar:

- `src/protocol/parse_events.py`
- `src/protocol/build_labels.py`
- `src/protocol/assign_phases.py`
- `src/protocol/map_emotions.py`

Estas funciones deben responder:

- qué evento es cuál fase;
- cómo mapear eventos de Unity a clases;
- cómo construir y validar las etiquetas por muestra.

### 3. Segmentación y preprocesado

Usar:

- `src/segmentation/preprocess.py`
- `src/segmentation/segment_signal.py`
- `src/segmentation/temporal_windowing.py`

Estas deben ser la base para cualquier notebook de análisis visual o diagnóstico del EMG.

### 4. Features y dataset

Usar:

- `src/features/extract_features.py`
- `src/features/statistical.py`
- `src/features/frequency.py`
- `src/features/temporal.py`
- `src/dataset/build_dataset.py`
- `src/dataset/export_dataset.py`
- `src/dataset/split_dataset.py`

### 5. Diagnóstico y reportes

Usar:

- `src/analysis/session_report.py`
- `src/analysis/dataset_report.py`
- `src/analysis/signal_diagnostics.py`

Estos son los módulos más relevantes para generar análisis explicativos de `data/analysis/`.

### 6. Modelos y evaluación

Usar:

- `src/models/*`
- `src/evaluation/*`

Aunque no todos los notebooks deben usar modelos, sí deben poder interpretar resultados sin reescribir la lógica de comparación.

---

## Requisitos de estilo y reproducibilidad

### Buenas prácticas para todos los notebooks

- Colocar al inicio una celda markdown con el objetivo.
- Decidir siempre la semilla y la configuración.
- No depender de outputs previos, salvo que se indique explícitamente.
- Usar variables claras (`session_dir`, `channels`, `fs`, `label_order`).
- Documentar supuestos de datos.
- Si se usa un filtro, mostrar claramente su configuración.
- Si hay una ventana temporal, mostrar exactamente el rango usado.
- El contenido del notebook debe permitir que alguien más lo entienda sin ver el código fuente completo.

### Mapa de decisiones esperadas

Cada notebook debe responder preguntas como:

- ¿qué estoy mirando?
- ¿de dónde vienen los datos?
- ¿qué está sincronizado?
- ¿qué estoy comparando?
- ¿qué patrón visual o cuantitativo veo?
- ¿qué significa eso para el problema?

---

## Notebook mínimo obligatorio por tipo

### 1. Notebook de EDA

Debe responder:

- ¿qué aspecto tiene la sesión completa?
- ¿qué canales están más activos?
- ¿hay ruido, artefactos o discontinuidades?
- ¿se observan eventos de Unity sobre la señal?

Contenido recomendado:

- carga de sesión;
- resumen detallado de amplitud, duración y sincronización;
- EMG completo con eventos de Unity;
- comparación raw vs filtered;
- al menos una visualización por canal;
- conclusión sobre calidad de la sesión.

### 2. Notebook de validación de etiquetas

Debe responder:

- ¿las etiquetas de Unity representan lo que ocurre en la señal?
- ¿hay fijación de la fase correcta?
- ¿hay desplazamiento de activación?

Contenido recomendado:

- resumen de eventos por fase;
- validación de trial;
- superposición etiqueta + señal;
- latencia, duración y consistencia por gesto;
- comparación por sujeto o repetición.

### 3. Notebook de sincronización

Debe responder:

- ¿los tiempos entre EMG, eventos y IMU coinciden?
- ¿hay drift o desfase?
- ¿qué rango está bien sincronizado?

Contenido recomendado:

- comparación de `t_ns` vs tiempos relativos;
- validación de offsets;
- visualización del registro de eventos frente a la señal;
- chequeo de huecos o duplicados.

### 4. Notebook de frecuencia

Debe responder:

- ¿qué bandas dominan la señal?
- ¿hay diferencia de energía por gesto o por canal?
- ¿cómo cambia el espectro entre condiciones?

Contenido recomendado:

- FFT/PSD por canal;
- potencia por banda;
- comparación de gesto vs reposo;
- análisis de frecuencia por sujeto o sesión.

### 5. Notebook de dataset

Debe responder:

- ¿cuántas muestras, segmentos o ventanas hay?
- ¿está balanceado el dataset?
- ¿hay clases muy poco representadas?

Contenido recomendado:

- distribución de clases;
- duración por segmento;
- cantidad de ventanas por sujeto;
- ratio de validación y train/test;
- diagnóstico de pérdida de señal.

### 6. Notebook de comparación de modelos

Debe responder:

- ¿qué modelo generaliza mejor?
- ¿cuál tiene mayor riesgo de fuga?
- ¿qué clases se confunden?

Contenido recomendado:

- tabla comparativa;
- matrices de confusión;
- métricas macro-F1, F1 por clase y balanced accuracy;
- brecha entre LOSO y K-Fold con fuga;
- interpretación por sujeto y por gesto.

---

## Análisis extra que conviene hacer además de lo que ya tengo en `data/analysis/`

Esto es importante para explicar mejor el proyecto y no quedarse solo en resultados puntuales. Se recomienda incorporar notebooks adicionales para:

### A. Diagnóstico por sujeto

- comparar cada sujeto por clase;
- observar qué sujeto es “más difícil” y por qué;
- analizar si el problema es un canal, un gesto o una sesión.

### B. Diagnóstico por canal

- calcular activación median, RMS, envelope y band power por canal;
- ver qué canales aportan más discriminación; 
- detectar si el problema está concentrado en EMG1/EMG2 o en EMG3/EMG4.

### C. Análisis temporal por gesto

- construir curvas de activación tras `hold_start` y `rest_start`;
- calcular latencia media y correlación con la fase real;
- identificar si la señal se comporta distinto antes y después del gesto.

### D. Análisis espectral por clase

- comparar PSD entre reposo, gesto y período de cuenta atrás;
- buscar si el gesto genera una banda dominante característica;
- identificar si la energía de frecuencia es informativa o si se parece demasiado al reposo.

### E. Fuga de información en datasets

- comparar LOSO vs K-Fold estratificado;
- cuantificar cuánto se infla el rendimiento cuando se mezclan ventanas del mismo segmento;
- dejar claro que las métricas que parecen muy altas pueden ser engañosas.

### F. Análisis de errores

- construir una tabla de errores por clase y sujeto;
- inspeccionar ejemplos de confusión;
- ligar los errores a patrones de amplitud, ruido o segmentos contaminados.

### G. Reporte narrativo de sesión

- convertir el contenido del notebook en un resumen legible;
- documentar “qué pasó en esta sesión” con un mini reporte de calidad.

---

## Qué debe servir este enfoque para `data/analysis/`

Los notebooks deberían funcionar como explicación del contenido que ya está generado en `data/analysis/` y no solo como código de prueba. En concreto, deben servir para:

- explicar qué hay dentro de `data/analysis/reports/`;
- contextualizar los resultados de modelos y sesiones;
- documentar qué archivos representan señales, etiquetas, evaluaciones y diagnósticos;
- justificar por qué una clase es difícil o por qué una sesión no es confiable;
- crear una narrativa reproducible para un informe técnico o presentación.

Por ejemplo, un notebook de reporting puede tomar:

- reportes de sesión;
- confusión por modelo;
- matrices de error;
- gráficos de frecuencia;
- estadísticas por sujeto;

y convertirlos en una explicación coherente: “la clase Triste extrae una activación débil y no diferenciada; la clase Reposo se identifica bien; la confusión principal ocurre entre X e Y”.

---

## Recomendación final

El proyecto necesita una convención clara: cada notebook debe responder una pregunta concreta, reutilizar las funciones del pipeline y aportar una pieza de análisis reproducible. La intención no es llenar la carpeta de notebooks con gráficos aislados, sino construir una narrativa completa de:

- calidad de los datos;
- sincronización temporal;
- validación de etiquetas;
- comportamiento espectral;
- rendimiento real de clasificación;
- explicación de los resultados en `data/analysis/`.

Si cada notebook sigue esta estructura, el proyecto será mucho más fácil de mantener, explicar y presentar a terceros.
