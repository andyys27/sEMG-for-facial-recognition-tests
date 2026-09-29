# Plan de reestructuración del repositorio

## Objetivo

Reorganizar el proyecto para que el flujo de trabajo quede claro, reproducible y basado en el protocolo experimental de Unity como fuente de verdad para las etiquetas.

La idea central es esta:

- La señal EMG es la materia prima.
- Los eventos de Unity describen el experimento y definen el protocolo.
- La segmentación y los filtros son herramientas de validación y refinamiento.
- La etiqueta final debe nacer del protocolo experimental, no de histeresis ni de reglas visuales.

---

## Principio de diseño

El repositorio debe separarse por responsabilidad funcional:

- carga de datos
- sincronización y validación
- protocolo y etiquetado
- segmentación
- extracción de features
- armado del dataset
- entrenamiento
- evaluación
- diagnóstico

Esto evita mezclar lógica de análisis, experimentación y entrenamiento en un mismo sitio.

---

## Estructura propuesta del repositorio

```text
sEMG-for-facial-recognition-tests/
├── README.md
├── requirements.txt
├── repo-structure-plan.md
├── .gitignore
├── .gitattributes
│
├── data/
│   ├── raw/
│   │   ├── subject_01/
│   │   │   ├── session_01/
│   │   │   │   ├── Baiobit_with_timestamp.csv
│   │   │   │   ├── FREEEMG_EMG_with_timestamp.csv
│   │   │   │   ├── Unity_events.csv
│   │   │   │   └── sesion01_meta.json
│   │   │   └── ...
│   │   └── subject_02/
│   │
│   ├── interim/
│   │   ├── subject_01/
│   │   └── subject_02/
│   │
│   ├── processed/
│   │   ├── labels/
│   │   ├── segments/
│   │   └── datasets/
│   │
│   └── analysis/
│       ├── subject_01/
│       ├── subject_02/
│       └── reports/
│
├── config/
│   ├── experiment/
│   │   ├── settings.yaml
│   │   └── labels.yaml
│   ├── protocol/
│   │   ├── events.yaml
│   │   └── phases.yaml
│   └── dataset/
│       ├── windows.yaml
│       ├── split.yaml
│       └── classes.yaml
│
├── src/
│   ├── io/
│   │   ├── __init__.py
│   │   ├── load_emg.py
│   │   ├── load_events.py
│   │   ├── load_metadata.py
│   │   └── sync_timestamps.py
│   │
│   ├── protocol/
│   │   ├── __init__.py
│   │   ├── parse_events.py
│   │   ├── assign_phases.py
│   │   ├── map_emotions.py
│   │   └── build_labels.py
│   │
│   ├── segmentation/
│   │   ├── __init__.py
│   │   ├── segment_signal.py
│   │   ├── validate_segment.py
│   │   └── temporal_windowing.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── temporal.py
│   │   ├── frequency.py
│   │   ├── statistical.py
│   │   └── extract_features.py
│   │
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── build_dataset.py
│   │   ├── split_dataset.py
│   │   └── export_dataset.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_models.py
│   │   ├── cnn_model.py
│   │   ├── train.py
│   │   └── save_model.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── confusion_matrix.py
│   │   ├── compare_models.py
│   │   └── diagnose_generalization.py
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── run_pipeline.py
│       └── orchestration.py
│
├── notebooks/
│   ├── eda/
│   ├── label_validation/
│   └── model_comparison/
│
├── scripts/
│   ├── prepare_data.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── run_full_pipeline.py
│
├── tests/
│   ├── protocol/
│   │   ├── test_parse_events.py
│   │   └── test_assign_phases.py
│   ├── synchronization/
│   │   └── test_sync_timestamps.py
│   ├── segmentation/
│   │   └── test_segment_signal.py
│   └── dataset/
│       └── test_build_dataset.py
│
├── main/
│   ├── legacy/
│   │   └── old_scripts/
│   ├── pipeline.py
│   ├── run_segmentation.py
│   ├── train_models.py
│   ├── compare_models.py
│   └── diagnose_generalization.py
│
└── README.md
```

---

## Qué va en cada parte

### 1) data/
La carpeta de datos debe ser la evidencia experimental y estar separada del código.

- raw/: datos originales, no modificados.
- interim/: datos ya procesados, pero todavía no listos para modelado.
- processed/: datos finales para análisis o entrenamiento.
- analysis/: resultados, experimentos y reportes visuales.

La regla clave: no se debe editar la data raw una vez registrada.

### 2) config/
La configuración debe describir el experimento y no depender del código.

- settings.yaml: fs, márgenes, rutas, labels generales.
- events.yaml: definición de eventos del protocolo de Unity.
- phases.yaml: qué define reposo, countdown y gesto.
- windows.yaml: tamaño de ventana, solapamiento, frecuencia.

### 3) src/io/
Es la capa de entrada/salida.

Responsabilidades:
- leer CSVs de EMG
- leer Unity_events.csv
- leer JSON de sesión
- normalizar columnas
- sincronizar timestamps
- validar consistencia entre fuentes

Esta capa no define etiquetas; solo transforma archivos a estructuras utilizable por el siguiente módulo.

### 4) src/protocol/
Es la parte más importante del refactor.

Responsabilidades:
- parsear eventos de Unity
- asociar cada evento con una emoción
- calcular bloques repetidos
- construir la secuencia fases: reposo, countdown, gesto
- generar label final por muestra o por ventana

Aquí es donde se resuelve la lógica de:
- cuándo empieza un gesto
- qué emoción corresponde a ese gesto
- qué bloque o repetición es
- qué intervalo corresponde a countdown

### 5) src/segmentation/
Aquí se toma el protocolo ya definido y se segmenta la señal.

Responsabilidades:
- recortar ventanas de tiempo
- validar si un segmento está completo
- filtrar segmentos inválidos
- preparar la señal para extracción de features

Importante: la segmentación debe apoyar al protocolo, no reemplazarlo.

### 6) src/features/
Extrae la representación matemática de cada segmento.

Ejemplos:
- RMS
- MAV
- ZCR
- entropía
- FFT
- bandas de frecuencia
- estadísticos por canal

### 7) src/dataset/
Construye el dataset final para aprendizaje.

Responsabilidades:
- juntar X y y
- manejar clases
- construir splits por sujeto o sesión
- exportar data para entrenamiento
- garantizar esquema consistente

### 8) src/models/
Entrena y guarda los modelos.

Responsabilidades:
- baseline clásico
- CNN
- entrenamiento
- validación interna
- serialización del modelo

### 9) src/evaluation/
Se encarga del análisis científico del rendimiento.

Responsabilidades:
- métricas de clasificación
- comparación entre modelos
- matrices de confusión
- diagnóstico de generalización
- análisis por sujeto

### 10) src/pipeline/
Orquestación general del proyecto.

Responsabilidades:
- encadenar fases del flujo
- ejecutar pipeline completo
- equilibrar reproducibilidad
- guardar artefactos

---

## Plan concreto de la Fase 1: estructura base y módulos iniciales

### Objetivo

Dejar el repositorio con una base limpia, legible y modular antes de tocar ninguna lógica de etiquetado ni de modelos.

### Entregable de esta fase

- estructura de carpetas creada
- paquetes vacíos o con __init__.py listos
- convenciones de nombres claras
- una primera organización de módulos por responsabilidad

### Orden recomendado de creación

#### 1. Estructura principal del repositorio

Crear primero estas carpetas:

- data/
- data/raw/
- data/interim/
- data/processed/
- data/analysis/
- config/
- config/experiment/
- config/protocol/
- config/dataset/
- src/
- src/io/
- src/protocol/
- src/segmentation/
- src/features/
- src/dataset/
- src/models/
- src/evaluation/
- src/pipeline/
- tests/
- tests/protocol/
- tests/synchronization/
- tests/segmentation/
- tests/dataset/

Esto es la base física. Si dejas esto hecho primero, todos los siguientes pasos encajan mejor.

#### 2. Archivos base de paquetes

Dentro de cada carpeta de src, crear un archivo __init__.py para convertir cada módulo en paquete Python.

Esto permite importar de forma ordenada después:

- src/io/__init__.py
- src/protocol/__init__.py
- src/segmentation/__init__.py
- src/features/__init__.py
- src/dataset/__init__.py
- src/models/__init__.py
- src/evaluation/__init__.py
- src/pipeline/__init__.py

#### 3. Crear la capa de entrada de datos

Antes que nada, crear los módulos que leen archivos y convierten datos a estructuras estándar:

- src/io/load_emg.py
- src/io/load_events.py
- src/io/load_metadata.py
- src/io/sync_timestamps.py

Su objetivo no es etiquetar, sino preparar la sesión para que luego la capa de protocolo pueda trabajar con ella.

#### 4. Crear la capa de protocolo

Después de io, crear los módulos de etiqueta y protocolo:

- src/protocol/parse_events.py
- src/protocol/assign_phases.py
- src/protocol/map_emotions.py
- src/protocol/build_labels.py

Estos serán los primeros módulos con lógica real de negocio. Aquí es donde se define el origen de las etiquetas.

#### 5. Crear los módulos de preparación del segmento

Después, crear los archivos para recortes y validación:

- src/segmentation/segment_signal.py
- src/segmentation/validate_segment.py
- src/segmentation/temporal_windowing.py

No hace falta todavía modelar todo el pipeline, pero sí dejar estos pasos claramente separados.

#### 6. Crear el esquema base para features y dataset

Luego crear los módulos iniciales, aunque todavía sin contenido avanzado:

- src/features/extract_features.py
- src/dataset/build_dataset.py
- src/dataset/split_dataset.py
- src/dataset/export_dataset.py

Estos módulos se completarán después, pero ya deben existir para no mezclar lógica más adelante.

#### 7. Crear la base de modelos y evaluación

Después de dataset, dejar preparados los puntos de entrada para modelos:

- src/models/train.py
- src/models/save_model.py
- src/evaluation/metrics.py
- src/evaluation/compare_models.py
- src/evaluation/diagnose_generalization.py

#### 8. Crear la orquestación general

Por último, definir la capa que coordina todo:

- src/pipeline/run_pipeline.py
- src/pipeline/orchestration.py

Esta capa no hace trabajo técnico pesado; solo ordena la secuencia del flujo completo.

### Criterio de éxito de la Fase 1

La fase termina cuando:

- cada pieza del proyecto tiene un hogar claro
- los nombres son consistentes
- no se mezclan datos, configuración, modelado y análisis
- puedes abrir una sesión desde io y pasarla a protocol sin que haya dependencias circularmente confusas

### Regla de trabajo para esta fase

No implementes aún:

- modelos complejos
- CNNs
- validación final del proyecto completo
- refinamientos de segmentación avanzados

Sí deberías hacer:

- la estructura
- los módulos base
- la separación clara del flujo
- la definición de la capa protocol como primera lógica funcional

---

## Fases recomendadas para trabajar en código

### Fase 1: base del repositorio y estructura
Objetivo: dejar el proyecto ordenado y consistente antes de tocar lógica.

Tareas:
- crear estructura de carpetas
- mover scripts viejos a una zona legacy si hace falta
- dejar carpetas de data y config
- definir módulos base en src

Resultado esperado:
- el repo se vea organizado
- cada responsabilidad tenga un sitio claro

### Fase 2: entrada de datos y carga
Objetivo: poder leer y validar los archivos sin depender de scripts ad hoc.

Tareas:
- cargar EMG raw
- cargar Unity_events.csv
- cargar metadata JSON
- normalizar columnas y tiempos
- sincronizar timestamps

Resultado esperado:
- poder abrir una sesión como una estructura estándar del proyecto

### Fase 3: protocolo y etiquetado
Objetivo: construir labels a partir del experimento y no de heurísticas visuales.

Tareas:
- parsear eventos de Unity
- mapear eventos a fases
- definir bloques y emociones
- crear dataframe etiquetado por tiempo
- exportar labels a processed/

Resultado esperado:
- cada muestra tiene etiqueta, emoción, bloque y fase

### Fase 4: segmentación
Objetivo: transformar la señal continua en ventanas útiles para clasificación.

Tareas:
- recortar según margen de protocolo
- validar duración
- preparar segmentos por gesto / descanso
- generar dataset provisional

Resultado esperado:
- cada ejemplo queda bien delimitado para entrenamiento

### Fase 5: features y dataset
Objetivo: convertir segmentos a representaciones aprendibles.

Tareas:
- extraer features por canal
- preparar X y y
- crear splits de entrenamiento
- guardar dataset limpio

Resultado esperado:
- dataset listo para modelos

### Fase 6: models y evaluación
Objetivo: entrenar, comparar y analizar model performance.

Tareas:
- entrenar baseline
- entrenar CNN
- comparar resultados
- diagnosticar generalización
- guardar reportes

Resultado esperado:
- métricas reproducibles y análisis claros

### Fase 7: refinamiento y estabilidad
Objetivo: estabilizar el pipeline para que pueda correr de manera reproducible.

Tareas:
- validación de pipeline completo
- configuración centralizada
- nombres consistentes
- limpieza de scripts heredados

Resultado esperado:
- proyecto robusto y mantenible

---

## Recomendación de trabajo incremental

Lo mejor es no intentar reestructurar todo de golpe.

La secuencia más sensata es:

1. estructura base
2. entrada de datos
3. protocolo
4. segmentación
5. features
6. dataset
7. modelos
8. evaluación

Cada fase debe dejar un estado funcional antes de pasar a la siguiente.

---

## Resumen ejecutivo

La reestructuración correcta del repositorio no consiste en “mover archivos”, sino en separar cuatro grandes ideas:

- la fuente del experimento
- la preparación de datos
- la definición de labels
- la generación del modelo

La clave de este proyecto es que Unity da la verdad del experimento, y la señal solo aporta evidencia física.

Cuando esa separación esté clara, el código será mucho más simple y mucho más confiable.

---

## Siguiente paso recomendado

Empieza por esta secuencia concreta:

1. crear carpetas data/, config/, src/
2. crear src/io/
3. crear src/protocol/
4. implementar lectura de Unity_events.csv
5. implementar mapeo de fases y labels
6. verificar con un ejemplo real de una sesión

Esto te permitirá avanzar de forma ordenada y sin volver a mezclar responsabilidades.
