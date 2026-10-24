# sEMG-for-facial-recognition-tests

Pipeline de extremo a extremo para clasificar **expresiones faciales** a partir de señales de **electromiografía de superficie (sEMG)** de 4 canales, evaluado con **validación cruzada Leave-One-Subject-Out (LOSO)** para medir qué tan bien generaliza cada modelo a una persona no vista durante el entrenamiento.

Las etiquetas **no se asignan a mano ni por umbrales**: nacen del protocolo experimental que registra Unity (`Unity_events.csv`). La señal EMG aporta la evidencia física; el protocolo define la verdad.

---

## Objetivo

Clasificar 5 estados a partir de 4 canales sEMG faciales:

| Clase | Origen de la etiqueta |
|---|---|
| **Reposo** | Intervalos `rest_start` del protocolo |
| **Sonrisa**, **Triste**, **Disgusto**, **Sorprendido** | Intervalos `hold_start` del protocolo (el gesto que Unity indica) |

La cuenta regresiva previa a cada gesto (`Countdown`) se registra pero **se excluye** del entrenamiento.

El eje central del proyecto no es solo entrenar clasificadores, sino **medir su capacidad de generalizar entre sujetos** sin la fuga de información que infla un K-Fold tradicional.

---

## Principios de diseño

1. **Unity es la fuente de verdad de las etiquetas.** Ni la segmentación ni los filtros deciden qué gesto es cada tramo.
2. **La segmentación apoya al protocolo, no lo reemplaza.** Recorta con márgenes calibrados y valida cada segmento (duración, huecos, NaN, canal plano).
3. **Evaluación sin fuga.** Las ventanas se solapan 50 %, así que nunca se separan ventanas de un mismo segmento entre entrenamiento y prueba.
4. **Separación por responsabilidad.** Carga, protocolo, segmentación, features, dataset, modelos y evaluación viven en paquetes distintos, con configuración en YAML y sin rutas ni parámetros incrustados en el código.
5. **Datos crudos intocables.** `data/raw/` no se modifica; todo lo derivado se puede regenerar.

---

## Pipeline

```
data/raw/<sujeto>/<sesión>/
  FREEEMG_EMG_with_timestamp.csv   Unity_events.csv   (Baiobit_*.csv, *meta.json opcionales)
        │
        ▼
1. Carga y sincronización                  src/io/
   - Reloj común t_ns (ns desde epoch) para EMG y eventos de Unity
   - Validación: monotonía, huecos, solapamiento EMG/Unity, desfase de reloj
        │
        ▼
2. Protocolo y etiquetado                  src/protocol/
   - Eventos de Unity → fases (Reposo / Countdown / Gesto) → etiqueta por muestra
   - Salida: data/processed/labels/
        │
        ▼
3. Preprocesamiento y segmentación         src/segmentation/
   - Señal continua: remoción de DC, notch 60 Hz, pasa-banda Butterworth 20–450 Hz (fase cero)
   - Señal filtrada guardada en data/interim/ (se reutiliza si la configuración no cambia)
   - Recorte por intervalo con márgenes, validación y ventaneo (200 ms, 50 % de solapamiento)
        │
        ▼
4. Features y dataset                      src/features/, src/dataset/
   - 70 features por ventana + ventanas crudas alineadas (para la CNN)
   - Splits sin fuga: LOSO, o leave-one-rep-out si hay un solo sujeto
        │
        ▼
5. Entrenamiento y evaluación              src/models/, src/evaluation/
   - RF, SVM, KNN, LDA, MLP, CNN 1D y una referencia de azar (dummy)
   - Comparación, matrices de confusión y diagnóstico de generalización
        │
        ▼
6. Diagnóstico de datos                    src/analysis/  →  data/analysis/
   - Gráficas y reportes de calidad por sesión y por dataset
```

---

## Formato de los datos de entrada

Cada sesión vive en `data/raw/<sujeto>/<sesión>/`:

```
data/raw/subject_01/session_01/
├── FREEEMG_EMG_with_timestamp.csv     # Timestamp,EMG1,EMG2,EMG3,EMG4
├── Unity_events.csv                   # eventos del protocolo (ver abajo)
├── Baiobit_with_timestamp.csv         # opcional: IMU
└── sesion01_meta.json                 # opcional
```

`Unity_events.csv` tiene las columnas `Timestamp, t_ns, source, t_rx_ns, t_unity_ns, offset_ns, seq, label, detail`. La columna `t_ns` ya está corregida al reloj del PC y es la que se usa para etiquetar; `detail` trae `trial`, `rep`, `gesture_idx` y `gesture`.

El protocolo registrado en Unity (`sequence_start`) define repeticiones, gestos, y las duraciones de reposo inicial, cuenta regresiva, gesto y reposo. La correspondencia evento → fase está en `config/protocol/events.yaml`:

| Evento de Unity | Fase |
|---|---|
| `rest_start` | Reposo |
| `gesture_show`, `countdown_3/2/1` | Countdown (excluida del entrenamiento) |
| `hold_start` | Gesto (etiqueta = nombre del gesto) |
| `session_start`, `sequence_start`, `sequence_end` | Estructurales: cierran la fase anterior |

> El reposo posterior a un gesto conserva el nombre del gesto en el `detail` de Unity, pero su etiqueta siempre es `Reposo`.

Si Unity emite un evento que no está en `events.yaml`, el pipeline lo avisa con un warning en lugar de ignorarlo en silencio.

---

## Preprocesamiento y segmentación

| Parámetro | Valor | Dónde se cambia |
|---|---|---|
| Remoción de DC | Sí | `config/experiment/settings.yaml` |
| Filtro notch | 60 Hz, Q = 30 | `settings.yaml` |
| Pasa-banda | Butterworth orden 4, 20–450 Hz, fase cero (`sosfiltfilt`) | `settings.yaml` |
| Márgenes de Gesto | descarta 0.5 s al inicio | `settings.yaml` → `margins` |
| Márgenes de Reposo | descarta 2.25 s al inicio y 0.5 s al final | `settings.yaml` → `margins` |
| Ventana | 200 ms, 50 % de solapamiento | `config/dataset/windows.yaml` |
| Segmento mínimo | 500 ms | `windows.yaml` |

Los márgenes se calibraron con los datos: `analyze_session` genera un perfil de activación alineado a `hold_start` y a `rest_start`. En el gesto, la activación se establece unos 0.5 s después del inicio; en el reposo, la relajación del gesto anterior tarda ~2.25 s en volver al nivel basal, y recortar menos contaminaba la clase Reposo con residuo muscular.

> Los filtros de fase cero son **no causales**: son válidos para clasificación offline, no para tiempo real.

---

## Features (70 por ventana)

17 por canal × 4 canales, más 2 features cruzadas entre grupos musculares:

| Grupo | Features | Descripción |
|---|---|---|
| Dominio del tiempo | `MAV`, `RMS`, `WL`, `VAR`, `IEMG`, `ZC`, `SSC` | Estándar en clasificación de gestos EMG (Hudgins et al. 1993; Phinyomark et al. 2012). `ZC` y `SSC` usan un umbral de amplitud (`zc_ssc_threshold`) |
| Dominio de la frecuencia | `MNF`, `MDF`, `BP_low`, `BP_mid`, `BP_high` | Frecuencia media y mediana, y potencia relativa en 20–60, 60–150 y 150–450 Hz |
| Envolvente | `env_mean`, `env_max`, `env_std` | Estadísticos de la envolvente RMS móvil (50 ms) dentro de la ventana |
| Z-score | `z_mean`, `z_max` | Envolvente normalizada con la media/desviación de la sesión (sin usar etiquetas) |
| Cruzadas entre grupos | `ratio_G12_G34`, `diff_G12_G34` | Cociente y diferencia del RMS de grupo; capturan el reclutamiento diferencial entre grupos musculares |

Los grupos (`G12` = EMG1+EMG2, `G34` = EMG3+EMG4) se definen en `config/dataset/features.yaml` y **deben ajustarse a la colocación de electrodos** de cada montaje. La CNN usa en cambio la forma de onda cruda de las ventanas, sin features manuales.

---

## Estrategia de evaluación

- **Splits** (`config/dataset/split.yaml`, `strategy: auto`):
  - **LOSO** si hay 2 o más sujetos: cada fold prueba con un sujeto completo y entrena con los demás.
  - **Leave-one-rep-out** si hay un solo sujeto: cada fold prueba con una repetición completa del protocolo. El reposo inicial siempre va a entrenamiento.
- **Balanceo solo en entrenamiento:** Reposo es ~4× más frecuente que cada gesto, así que se recorta cada clase a la mediana de tamaños dentro del entrenamiento de cada fold. El test conserva su distribución real.
- **Hiperparámetros fijos** (`config/experiment/models.yaml`): no hay búsqueda de hiperparámetros, porque ajustarlos con los mismos folds con que se reporta inflaría las métricas.
- **Métrica principal: macro-F1**, no accuracy, por el desbalance de clases. También se reportan balanced accuracy, accuracy y weighted-F1.
- **Métrica por segmento:** la clase más votada entre las ventanas de un segmento. Responde "¿se acierta el gesto completo?", que es lo que usaría una aplicación real.
- **Referencia de azar** (`dummy`): cualquier modelo que no la supere claramente no aprendió nada.

### Modelos

| Tipo | Modelos |
|---|---|
| Machine Learning tradicional | Random Forest (RF), SVM, K-Nearest Neighbors (KNN), Linear Discriminant Analysis (LDA), Multilayer Perceptron (MLP) |
| Deep Learning | CNN 1D sobre la señal cruda (opcional, requiere TensorFlow). Valida con segmentos completos y normaliza con `arcsinh(x / s)` |
| Referencia | `dummy` (clasificador aleatorio estratificado) |

Todos los modelos tabulares van en un `Pipeline` con `StandardScaler`, ajustado solo con el entrenamiento de cada fold.

---

## Resultados

**Configuración:** LOSO con **2 sujetos** (`subject_01`, `subject_02`), 12 104 ventanas, 70 features. Cada fold entrena con **un solo sujeto** y prueba con el otro, que es el caso más exigente para generalizar. Resultados generados por `python -m scripts.evaluate_models` (`data/analysis/reports/model_comparison.md`).

### Comparación de modelos

| Modelo | Macro-F1 (agregado) | Macro-F1 por fold (media ± desv.) | Balanced acc. | Accuracy | Macro-F1 por segmento | Accuracy por segmento |
|---|---|---|---|---|---|---|
| MLP | **0.548** | 0.496 ± 0.035 | 0.561 | 0.675 | 0.568 | 0.734 |
| SVM | 0.541 | 0.469 ± 0.084 | 0.554 | 0.658 | 0.547 | 0.696 |
| CNN 1D | 0.531 | 0.493 ± 0.019 | 0.524 | 0.611 | 0.574 | 0.664 |
| KNN | 0.525 | 0.518 ± 0.011 | 0.548 | 0.641 | 0.529 | 0.681 |
| Random Forest | 0.510 | 0.498 ± 0.023 | 0.533 | 0.559 | 0.518 | 0.582 |
| LDA | 0.414 | 0.365 ± 0.106 | 0.456 | 0.526 | 0.421 | 0.576 |
| *Dummy (azar)* | *0.182* | *0.183 ± 0.001* | *0.200* | *0.201* | *0.163* | *0.199* |

### F1 por clase (agregado de ambos folds)

| Modelo | Reposo | Sonrisa | Triste | Disgusto | Sorprendido |
|---|---|---|---|---|---|
| MLP | 0.86 | 0.85 | 0.21 | 0.51 | 0.32 |
| SVM | 0.84 | 0.83 | 0.18 | 0.57 | 0.28 |
| KNN | 0.82 | 0.84 | 0.19 | 0.60 | 0.17 |
| CNN 1D | 0.74 | 0.82 | 0.36 | 0.51 | 0.23 |
| Random Forest | 0.66 | 0.88 | 0.06 | 0.61 | 0.34 |
| LDA | 0.72 | 0.74 | 0.34 | 0.21 | 0.06 |
| *Dummy* | *0.29* | *0.15* | *0.16* | *0.16* | *0.16* |

### Diagnóstico de generalización (evaluación honesta vs. K-Fold con fuga)

El script compara LOSO con un K-Fold estratificado por ventanas, que mezcla ventanas solapadas del mismo segmento entre entrenamiento y prueba:

| Modelo | Macro-F1 honesto (LOSO) | Macro-F1 con fuga | Brecha | Accuracy honesta | Accuracy con fuga |
|---|---|---|---|---|---|
| Random Forest | 0.510 | 0.915 | **+0.405** | 0.559 | 0.935 |
| SVM | 0.541 | 0.889 | **+0.349** | 0.658 | 0.914 |
| KNN | 0.525 | 0.860 | **+0.334** | 0.641 | 0.888 |

### Lectura de los resultados

- **Hay información que transfiere entre sujetos.** Todos los modelos superan por mucho la referencia de azar (macro-F1 ≈ 0.18), aun entrenando con una sola persona.
- **No se puede declarar un mejor modelo.** MLP, SVM, CNN, KNN y Random Forest quedan en 0.51–0.55 de macro-F1 agregado, dentro de la variación entre folds (hasta ± 0.084), y el orden cambia según se mire el agregado (MLP primero) o la media por fold (KNN primero). Solo LDA queda claramente por debajo.
- **Reposo y Sonrisa se identifican bien** (F1 ≈ 0.82–0.88 en los mejores modelos). **Disgusto** queda en nivel medio (≈ 0.5–0.6). **Triste y Sorprendido son las clases más difíciles** (F1 entre 0.06 y 0.36).
- **Hipótesis a verificar** (no confirmada en ambos sujetos): en el análisis de `subject_01`, Triste mostró una activación en los 4 canales cercana al nivel de reposo, y Disgusto y Sorprendido activaron los mismos canales (EMG1/EMG2). Eso explicaría que Triste se confunda con Reposo y que Disgusto y Sorprendido se confundan entre sí; las matrices de confusión por modelo (`data/analysis/models/<modelo>/confusion_matrix.png`) permiten comprobarlo.
- **La fuga infla el rendimiento entre 0.33 y 0.41 de macro-F1.** Un K-Fold por ventanas haría parecer a estos modelos casi perfectos (> 0.86), cuando con un sujeto nuevo rinden ~0.5. Por eso toda cifra de este proyecto se reporta con evaluación sin fuga.
- **Significancia estadística no disponible.** Con 2 folds, el menor p-valor posible de un Wilcoxon es 0.5; ninguna diferencia entre modelos puede ser significativa. Se necesitan al menos 6 folds (6 sujetos en LOSO) para que sea posible alcanzar p < 0.05.

> Estos resultados corresponden a una **prueba de concepto con 2 sujetos**. Sirven para validar el pipeline y detectar qué clases son difíciles, no como medida definitiva de desempeño.

---

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt            # dependencias base
├── requirements-cnn.txt        # TensorFlow (solo para la CNN)
├── pytest.ini
│
├── config/                     # toda la configuración, sin parámetros en el código
│   ├── experiment/             # settings.yaml (filtros, márgenes), labels.yaml, models.yaml
│   ├── protocol/               # events.yaml (evento → fase), phases.yaml
│   └── dataset/                # windows.yaml, features.yaml, classes.yaml, split.yaml
│
├── data/                       # ignorado por git (se versiona solo la estructura)
│   ├── raw/<sujeto>/<sesión>/  # datos originales, no se modifican
│   ├── interim/<sujeto>/<sesión>/   # señal filtrada, eventos parseados, reporte de sincronización
│   ├── processed/
│   │   ├── labels/             # intervalos y etiquetas por muestra
│   │   ├── segments/           # segmentos y ventanas por sesión
│   │   └── datasets/           # features_dataset.csv, raw_windows.npz
│   └── analysis/
│       ├── <sujeto>/<sesión>/  # overview, perfil de activación, calidad de la sesión
│       ├── models/<modelo>/    # métricas, predicciones, matriz de confusión, modelo
│       └── reports/            # comparación de modelos, diagnóstico, resumen del dataset
│
├── src/
│   ├── config.py               # carga de YAML
│   ├── io/                     # lectura, sincronización de relojes, datos intermedios
│   ├── protocol/               # eventos de Unity → fases → etiquetas
│   ├── segmentation/           # filtrado, recorte con márgenes, validación, ventaneo
│   ├── features/               # features temporales, de frecuencia y estadísticas
│   ├── dataset/                # armado del dataset, splits sin fuga, exportación
│   ├── models/                 # baselines (sklearn), CNN 1D, entrenamiento, serialización
│   ├── evaluation/             # métricas, matrices de confusión, comparación, diagnóstico
│   ├── analysis/               # gráficas y reportes de calidad de datos
│   └── pipeline/               # orquestación (run_pipeline: pendiente)
│
├── scripts/                    # puntos de entrada (CLI delgados)
│   ├── label_session.py        # etiqueta una sesión y resume el resultado
│   ├── segment_session.py      # segmenta una sesión
│   ├── analyze_session.py      # segmenta y genera diagnósticos de una sesión
│   ├── prepare_data.py         # todas las sesiones → dataset
│   ├── train_models.py         # entrena y evalúa modelos
│   ├── evaluate_models.py      # compara modelos y diagnostica la generalización
│   └── run_full_pipeline.py    # pendiente
│
├── tests/                      # pytest: protocolo, sincronización, segmentación, dataset, modelos
├── notebooks/                  # exploración (eda, label_validation, model_comparison)
└── main/legacy/                # pipeline anterior (etiquetado manual), solo como referencia
```

---

## Instalación

```bash
git clone https://github.com/andyys27/sEMG-for-facial-recognition-tests.git
cd sEMG-for-facial-recognition-tests
python -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-cnn.txt    # opcional: solo para la CNN 1D
```

Requiere Python 3.12 o superior. TensorFlow es opcional: si no está instalado, `train_models --cnn` omite la CNN y continúa con el resto.

Dependencias principales: `numpy`, `pandas`, `scipy` (señales y datos), `scikit-learn` y `joblib` (modelos tabulares), `matplotlib` (gráficas), `pyyaml` (configuración), `pytest`.

---

## Uso

Todos los scripts se ejecutan como módulos desde la raíz del repositorio.

### 1. Colocar los datos

```
data/raw/subject_01/session_01/{FREEEMG_EMG_with_timestamp.csv, Unity_events.csv}
```

### 2. Verificar una sesión (recomendado con cada sesión nueva)

```bash
python -m scripts.label_session data/raw/subject_01/session_01
python -m scripts.analyze_session data/raw/subject_01/session_01 --print
```

`label_session` muestra cuántos intervalos y muestras hay por etiqueta. `analyze_session` genera en `data/analysis/<sujeto>/<sesión>/`:

- `overview.png`: activación por canal con las fases sombreadas y los segmentos usados. Si el protocolo está bien sincronizado, la actividad de cada gesto cae dentro de su banda de color.
- `activation_profile.png/.csv`: perfil de activación por gesto, alineado a `hold_start` y `rest_start`, con la zona que conservan los márgenes. Sirve para calibrar `margins`.
- `activity_by_label.png/.csv`: activación media por etiqueta y canal.
- `quality_report.json`: conteos, comparación contra el protocolo y avisos (gestos esperados vs. encontrados, segmentos descartados, desbalance de clases, huecos en la señal).

### 3. Construir el dataset

```bash
python -m scripts.prepare_data
```

Procesa todas las sesiones de `data/raw/`, extrae features, exporta `features_dataset.csv` y `raw_windows.npz`, y genera `data/analysis/reports/` (`dataset_summary.md`, balance de clases y mapa de features por clase). Muestra también los folds de evaluación.

### 4. Entrenar y evaluar

```bash
python -m scripts.train_models --list              # modelos disponibles
python -m scripts.train_models                     # dummy, rf, svm, knn, lda, mlp
python -m scripts.train_models --models rf svm     # solo algunos
python -m scripts.train_models --cnn               # agrega la CNN (requiere TensorFlow)
```

### 5. Comparar modelos y diagnosticar la generalización

```bash
python -m scripts.evaluate_models                  # tabla, gráfica, significancia y diagnóstico
python -m scripts.evaluate_models --diagnose rf knn
python -m scripts.evaluate_models --no-diagnose
```

### 6. Correr las pruebas

```bash
python -m pytest
```

---

## Añadir un nuevo sujeto

1. Crea `data/raw/subject_NN/session_01/` con `FREEEMG_EMG_with_timestamp.csv` y `Unity_events.csv`.
2. Revisa la sesión con `scripts.analyze_session` y comprueba `overview.png` y `quality_report.json`.
3. Si el montaje de electrodos es distinto, ajusta `channel_groups` en `config/dataset/features.yaml`.
4. Ejecuta `scripts.prepare_data`, `scripts.train_models` y `scripts.evaluate_models`.

No hace falta editar código: la estrategia de evaluación pasa a LOSO automáticamente cuando hay 2 o más sujetos. Cuantos más sujetos se agreguen, más confiables serán las métricas y las pruebas de significancia.

---

## Reproducibilidad

- Semilla fija (`seed: 42` en `config/experiment/models.yaml`) para el balanceo, los modelos y la CNN.
- Hiperparámetros y parámetros de procesamiento en YAML, versionados con el código.
- La señal filtrada de `data/interim/` lleva una huella de la configuración y se recalcula sola si cambian los filtros o los datos de origen.
- Todo lo de `interim/`, `processed/` y `analysis/` se regenera desde `data/raw/`.

---

## Limitaciones y trabajo futuro

- **Solo 2 sujetos.** Las métricas son una prueba de concepto; las pruebas de significancia requieren al menos 6 sujetos para poder alcanzar p < 0.05.
- **Entrenamiento con un solo sujeto por fold** en la configuración actual; con más sujetos el rendimiento entre personas debería mejorar y estabilizarse.
- **Clases Triste y Sorprendido** poco separables con el montaje actual de 4 canales; conviene revisar la colocación de electrodos y la intensidad de los gestos.
- **Sin ajuste de hiperparámetros.** Un ajuste correcto requiere validación anidada para no sesgar la evaluación.
- **Filtrado offline** (fase cero): no es aplicable tal cual a una implementación en tiempo real.
- **Pendiente:** orquestador de un solo comando (`run_full_pipeline`), limpieza de `main/legacy/` y notebooks de exploración.

---

## Referencias

- Hudgins, B., Parker, P., & Scott, R. N. (1993). *A new strategy for multifunction myoelectric control*. IEEE Transactions on Biomedical Engineering.
- Phinyomark, A., Phukpattaranont, P., & Limsakul, C. (2012). *Feature reduction and selection for EMG signal classification*. Expert Systems with Applications.