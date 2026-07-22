# sEMG-for-facial-recognition-tests

Pipeline de extremo a extremo para clasificar **expresiones faciales** a partir de señales de **electromiografía de superficie (sEMG)**, evaluado con **validación cruzada Leave-One-Subject-Out (LOSO)** para medir qué tan bien generaliza cada modelo a sujetos no vistos durante el entrenamiento.

---

## 🎯 Objetivo del proyecto

Desarrollar y evaluar un pipeline completo de Machine Learning y Deep Learning que, a partir de señales sEMG de 4 canales faciales, sea capaz de clasificar 5 estados/expresiones:

- **Reposo** (estado base)
- **Sonrisa**
- **Triste**
- **Disgusto**
- **Sorprendido**

El eje central del proyecto no es solo entrenar clasificadores, sino **medir su capacidad de generalización entre sujetos distintos**, algo que un K-Fold tradicional tiende a sobreestimar de forma optimista.

---

## 🧠 ¿Por qué LOSO y no K-Fold tradicional?

En señales biológicas como el sEMG, cada persona tiene una "firma" fisiológica propia (fuerza muscular, ubicación de electrodos, impedancia de piel, etc.). Un K-Fold estratificado normal mezcla ventanas del mismo sujeto entre train y test, por lo que el modelo puede aprender a reconocer *a la persona* en vez del *gesto*, inflando artificialmente el rendimiento.

**Leave-One-Subject-Out (LOSO)** entrena con *N-1* sujetos y evalúa siempre sobre un sujeto completamente no visto, rotando el sujeto de prueba en cada fold. Esto da una estimación realista de cómo se comportaría el sistema con una persona nueva.

El script [`main/diagnose_generalization.py`](main/diagnose_generalization.py) compara explícitamente ambos esquemas (LOSO vs. K-Fold con fuga de sujeto) para diagnosticar si un F1 bajo se debe a un problema de generalización entre sujetos o a un problema de la señal/etiquetado en sí.

---

## 🔬 Pipeline del proyecto

```
Señal cruda sEMG (4 canales)
        │
        ▼
1. Preprocesamiento por canal        main/processing.py, main/pipeline.py, main/main.py
   - Remoción de offset DC
   - Filtro Notch (60 Hz) + Butterworth pasa-banda (20–450 Hz)
   - Envolvente RMS + normalización Z-score
        │
        ▼
2. Segmentación de eventos           main/emg_segmentation/, main/run_segmentation.py
   - Umbral doble (hysteresis) por grupo muscular, baseline fijo o móvil (mediana+MAD)
   - Consenso entre grupos, refinamiento de onset/offset, relleno de huecos (gap filling/rescue)
   - Exporta épocas individuales + CSV completo etiquetado (Reposo/Countdown/Emoción)
        │
        ▼
3. Construcción del dataset          main/build_dataset.py, main/emg_ml/dataset.py
   - Ventaneo con solapamiento (windowing + overlap)
   - Feature engineering por ventana y canal      main/emg_ml/features.py
   - Features crudas para CNN                     main/emg_ml/raw_dataset.py
        │
        ▼
4. Entrenamiento y evaluación (LOSO)
   - Modelos tabulares: RF, SVM, KNN, LDA, MLP     main/train_models.py
   - CNN 1D sobre la forma de onda cruda           main/train_cnn.py
        │
        ▼
5. Análisis y comparación de resultados
   - Tabla y gráficas comparativas entre modelos   main/compare_models.py
   - Significancia estadística (Wilcoxon+Holm)     main/compare_significance.py
   - LOSO vs. K-Fold con fuga de sujeto            main/diagnose_generalization.py
   - Análisis de errores por clase/sujeto          main/error_analysis.py
```

---

## 📊 Datos y metodología

### Clases de expresión facial (5 clases)
Reposo · Sonrisa · Triste · Disgusto · Sorprendido

### Extracción de características (feature engineering)
Sobre cada ventana y canal sEMG se calculan, entre otras:

| Grupo | Features | Descripción |
|---|---|---|
| Dominio del tiempo | `MAV`, `RMS`, `WL`, `VAR`, `ZC`, `SSC`, `IEMG` | Estándar en literatura de clasificación de gestos EMG (Hudgins et al. 1993; Phinyomark et al. 2012) |
| Envolvente | `_env_mean`, `_env_max`, `_env_std` | Estadísticos sobre la envolvente RMS de activación |
| Z-score | `_z_mean`, `_z_max` | Estadísticos sobre la envolvente normalizada |
| Cruzadas entre grupos | `ratio_GrupoA_GrupoB`, `diff_GrupoA_GrupoB` | Capturan el patrón de reclutamiento diferencial entre grupos musculares |

Para la CNN se usa en cambio la forma de onda cruda por canal (`EMG*_Filtered`), sin feature engineering manual.

### Estrategia de evaluación

- **Leave-One-Subject-Out (LOSO) Cross-Validation:** entrena con *N-1* sujetos, evalúa en el sujeto restante, rotando.
- **Escenarios de experimentación:**
  - `all_data`: usa todas las ventanas extraídas.
  - `high_confidence`: entrena solo con ventanas etiquetadas con alta confianza.

### Modelos evaluados

| Tipo | Modelos |
|---|---|
| Machine Learning tradicional | Random Forest (RF), Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Linear Discriminant Analysis (LDA), Multilayer Perceptron (MLP) |
| Deep Learning | Red Neuronal Convolucional 1D (CNN) sobre la señal cruda |

---

## 📈 Resultados (LOSO, escenario `all_data`)

| Modelo | Accuracy (agregado) | Macro F1 | Weighted F1 |
|---|---|---|---|
| **SVM** | **0.479** | **0.293** | **0.417** |
| Random Forest | 0.459 | 0.249 | 0.389 |
| MLP | 0.454 | 0.221 | 0.360 |
| LDA | 0.390 | 0.237 | 0.365 |
| KNN | 0.416 | 0.207 | 0.349 |
| CNN 1D | 0.329 | 0.234 | 0.338 |

> Métricas extraídas de `Results/<Modelo>_LOSO/all_data/metrics.json`. En todos los modelos, la clase **Reposo** es sistemáticamente la mejor identificada, mientras que **Sorprendido** es la más difícil de distinguir, lo que sugiere una fuerte superposición de patrones musculares entre esa expresión y las demás con la cantidad actual de sujetos.
>
> Estos números corresponden a un dataset con muy pocos sujetos (actualmente 2, `Test1` y `Test2`), por lo que deben leerse como una prueba de concepto del pipeline, no como una medida definitiva de desempeño. El script `compare_significance.py` avisa explícitamente cuando hay muy pocos folds para que un test de Wilcoxon sea concluyente.

Para regenerar y actualizar esta tabla junto con sus gráficas:

```bash
python -m main.compare_models
```

---

## 📁 Estructura del repositorio

```
.
├── main/
│   ├── emg_ml/                  # Feature engineering, datasets, modelos y CNN
│   │   ├── dataset.py           # Construcción del dataset de features tabular
│   │   ├── features.py          # Cálculo de features por ventana/canal
│   │   ├── model.py             # LOSO para modelos tabulares (sklearn)
│   │   ├── models.py            # Fábrica de modelos (RF, SVM, KNN, LDA, MLP)
│   │   ├── cnn_model.py         # CNN 1D + LOSO (Keras/TensorFlow)
│   │   ├── raw_dataset.py       # Dataset de ventanas crudas para la CNN
│   │   └── plots.py             # Generación de gráficas de resultados
│   ├── emg_segmentation/        # Detección de eventos y etiquetado de épocas
│   │   ├── config.py            # Configuración del pipeline de segmentación
│   │   ├── detection.py         # Detección de activaciones por canal/grupo
│   │   ├── epochs.py            # Extracción y exportación de épocas
│   │   ├── labeling.py          # Etiquetado del CSV completo (Reposo/Emoción)
│   │   └── plotting.py          # Gráficas de diagnóstico de segmentación
│   ├── processing.py            # Filtros (bandpass, notch), envolvente, FFT
│   ├── pipeline.py               # Pipeline de preprocesamiento por canal
│   ├── main.py                   # Punto de entrada: preprocesa señal cruda
│   ├── run_segmentation.py       # Punto de entrada: segmentación por sujeto
│   ├── build_dataset.py          # Punto de entrada: dataset de features
│   ├── train_models.py           # Punto de entrada: entrena modelos tabulares LOSO
│   ├── train_cnn.py              # Punto de entrada: entrena CNN LOSO
│   ├── compare_models.py         # Tabla y gráficas comparativas entre modelos
│   ├── compare_significance.py   # Test de significancia (Wilcoxon + Holm-Bonferroni)
│   ├── diagnose_generalization.py# LOSO vs. K-Fold con fuga de sujeto
│   └── error_analysis.py         # Análisis de errores por clase/sujeto
├── Results/                      # metrics.json por modelo y escenario
│   ├── CNN_LOSO/
│   ├── KNN_LOSO/
│   ├── LDA_LOSO/
│   ├── MLP_LOSO/
│   ├── RandomForest_LOSO/
│   └── SVM_LOSO/
├── Test1/                        # Datos y configuración de segmentación del sujeto 1
│   ├── config_test1.json
│   ├── Analysis/
│   ├── Active_Channels/
│   └── Data/
└── Test2/                        # Datos y configuración de segmentación del sujeto 2
    ├── config_test2.json
    ├── Analysis/
    ├── Active_Channels/
    └── Data/
```

> Las carpetas `Test1/` y `Test2/` representan cada una a un sujeto de prueba. Para añadir un nuevo sujeto, se agrega una carpeta análoga con su propio `Data/` y configuración de segmentación.

---

## ⚙️ Instalación

```bash
git clone https://github.com/andyys27/sEMG-for-facial-recognition-tests.git
cd sEMG-for-facial-recognition-tests
python -m venv .venv
source .venv/bin/activate      # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencias principales del proyecto:

- `numpy`, `pandas`, `scipy` — procesamiento de señales y manejo de datos
- `scikit-learn` — modelos tabulares (RF, SVM, KNN, LDA, MLP), métricas, LOSO
- `matplotlib` — visualizaciones y gráficas de diagnóstico
- `tensorflow` / `keras` — CNN 1D
- `statsmodels` / `scipy.stats` — pruebas de significancia estadística (Wilcoxon)

---

## 🚀 Uso

Todos los scripts están pensados para ejecutarse como módulos desde la raíz del repositorio (`python -m main.<script>`).

### 1. Preprocesar señal cruda por sujeto

```bash
cd main
python main.py
```

Filtra cada canal EMG (notch + bandpass), calcula la envolvente RMS y la normalización Z-score, y genera gráficas de diagnóstico por canal en `Analysis/`.

### 2. Segmentar eventos y etiquetar épocas

```bash
python -m main.run_segmentation
```

Detecta activaciones musculares por canal y grupo, arma el consenso entre grupos, refina los onsets/offsets y exporta las épocas individuales junto con un CSV completo etiquetado (`processed_labeled.csv`).

### 3. Construir el dataset de features

```bash
python -m main.build_dataset
```

Genera `Dataset/features_dataset.csv`, con una fila por ventana y las features de todos los sujetos configurados.

### 4. Entrenar los modelos tabulares (LOSO)

```bash
python -m main.train_models                         # todos los modelos, escenario all_data
python -m main.train_models --models rf svm         # solo modelos específicos
python -m main.train_models --scenario both         # corre all_data y high_confidence
python -m main.train_models --list                  # lista los modelos disponibles
```

### 5. Entrenar la CNN (LOSO)

```bash
python -m main.train_cnn
```

Construye (o carga desde caché) el dataset de ventanas crudas y entrena/evalúa la CNN 1D con LOSO completo.

### 6. Comparar modelos y analizar resultados

```bash
python -m main.compare_models            # tabla + gráficas comparativas
python -m main.compare_significance      # test de Wilcoxon + Holm-Bonferroni por pares
python -m main.diagnose_generalization   # LOSO vs. K-Fold con fuga de sujeto
python -m main.error_analysis            # análisis de errores por clase/sujeto
```

---

## 🧩 Añadir un nuevo sujeto

1. Crea una carpeta `TestN/Data/` con la señal cruda (`FREEEMG_EMG_with_timestamp.csv`).
2. Ejecuta el preprocesamiento (`main.py`) y la segmentación (`run_segmentation.py`) apuntando a esa carpeta, ajustando `channel_groups` y umbrales (`k_baseline_per_group`) según la calibración del sujeto.
3. Agrega el sujeto a `subject_specs` en `build_dataset.py` y `train_cnn.py`.
4. Vuelve a correr `build_dataset.py`, `train_models.py` y `train_cnn.py` para regenerar el dataset y las métricas LOSO con el nuevo sujeto incluido.

Cuantos más sujetos se agreguen, más confiables serán las métricas LOSO y las pruebas de significancia estadística.

---

## 📌 Notas y limitaciones

- El dataset actual incluye únicamente **2 sujetos** (`Test1`, `Test2`), por lo que los resultados reportados son una prueba de concepto del pipeline y no deben interpretarse como una medida definitiva de generalización.
- `diagnose_generalization.py` está pensado justamente para diferenciar si un F1 bajo en alguna clase se debe a variabilidad real entre sujetos (se necesitan más sujetos) o a un problema de la señal/etiquetado en sí (revisar segmentación o sincronización de eventos).
- Las pruebas de significancia estadística (`compare_significance.py`) deben tratarse como orientativas mientras el número de sujetos/folds sea bajo (<6).

---

## 📚 Referencias

- Hudgins, B., Parker, P., & Scott, R. N. (1993). *A new strategy for multifunction myoelectric control*. IEEE Transactions on Biomedical Engineering.
- Phinyomark, A., Phukpattaranont, P., & Limsakul, C. (2012). *Feature reduction and selection for EMG signal classification*. Expert Systems with Applications.
