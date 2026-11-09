# Protocolo experimental completo para adquisición, análisis y validación de señales sEMG faciales con protocolo de Unity y proyección a investigación clínica en parálisis facial

## 1. Título del estudio

Adquisición y análisis de señales electromiográficas de superficie (sEMG) faciales durante expresiones gestuales controladas con protocolo de Unity, con evaluación de generalización y proyección a aplicaciones clínicas en parálisis facial.

## 2. Responsable del estudio

- Investigador responsable: completar nombre completo, cargo y institución.
- Institución: completar universidad, centro o laboratorio.
- Contacto: completar correo electrónico y teléfono.
- Versión del protocolo: borrador completo.
- Fecha: 09/11/2026.

## 3. Resumen ejecutivo

Este estudio se propone desarrollar y validar un protocolo experimental de adquisición de señales electromiográficas de superficie (sEMG) de la musculatura facial para la clasificación y análisis de expresiones faciales en condiciones controladas. El objetivo principal es establecer una base metodológica reproducible para distinguir patrones de activación muscular asociados a gestos faciales, a partir de un protocolo temporal definido en Unity, con sincronización precisa con la señal EMG y con etiquetado basado en el protocolo experimental y no en inferencias visuales aisladas.

La etapa actual del estudio corresponde a una fase de validación y construcción de la base de datos. En esta fase, se usan 4 canales sEMG: 2 ubicados en la región frontal y 2 en la región de los cachetes, con colocación según la musculatura facial de interés. La señal se registra con el dispositivo FREEEMG, mientras que la información del movimiento y la postura corporal se obtienen con un IMU Baiobit localizado en la parte posterior de la correa del visor de VR, y el estímulo experimental se presenta mediante un visor Meta Quest 3. La temporalidad y la secuencia de la tarea se definen en Unity, que marca los instantes de reposo, cuenta regresiva y mantenimiento del gesto.

El estudio se diseña como un protocolo integral y escalable. La fase actual es un diseño con repeticiones fijas, secuencia estándar y datos suficientes para validar la señal, la sincronización, la segmentación y la trazabilidad del etiquetado. La extensión prevista incluye la acumulación de suficientes sujetos y sesiones para poder realizar análisis de robustez, generalización intersujeto, variación de condiciones experimentales y, en última instancia, un marco que pueda adaptarse a estudios clínicos orientados a la parálisis facial.

## 4. Justificación y antecedentes

La evaluación de la actividad muscular facial mediante sEMG es una herramienta útil para entender la ejecución de expresiones faciales, medir la activación de grupos musculares y cuantificar patrones de asimetría o disfunción. Esta tecnología es especialmente relevante en contextos donde la expresión facial es un indicador de función motora, de dolor, de fatigabilidad o de oclusión neuromuscular. En el caso de la parálisis facial, la asimetría y la diferencia de activación entre lados faciales son fundamentales para la evaluación y el seguimiento clínico.

Aunque existen protocolos para reconocimiento de expresiones faciales y para rehabilitación de la musculatura facial, la validación de una estrategia basada en sEMG para reconocer gestos faciales en condiciones controladas y reproducibles sigue siendo un punto clave. El enfoque de este estudio es ir más allá de una simple grabación de señales: la investigación centra la validez metodológica en la sincronización, en la definición temporal del protocolo y en la trazabilidad del etiquetado experimental. Esto es de especial importancia para estudios que luego puedan ser transferidos a escenarios clínicos.

La incorporación de VR mediante Meta Quest 3 introduces un entorno de estimulación visual y estructural que ayuda a controlar la atención del participante y a mantener la consistencia del protocolo. El IMU Baiobit en la parte posterior de la correa del visor permite complementar la información de postura y movimiento de la cabeza, facultando análisis de confusión entre activación muscular facial y movimiento de cabeza o desplazamiento postural.

## 5. Objetivo general

Desarrollar y validar un protocolo experimental completo de adquisición de señales sEMG faciales y su análisis para la clasificación de expresiones faciales en condiciones controladas, con una proyección explícita hacia la evaluación de la actividad muscular facial en contextos de parálisis facial y otras alteraciones de la motricidad facial.

## 6. Objetivos específicos

1. Registrar señales sEMG faciales durante secuencias experimentales estructuradas, controladas y repetibles.
2. Validar la sincronización entre la señal de EMG y los eventos temporales generados por Unity.
3. Construir un protocolo de etiquetado reproducible basado en la secuencia experimental y no en interpretación manual aislada.
4. Evaluar la capacidad de distinguir expresiones faciales, restos y activaciones musculares a partir de la señal registrada.
5. Analizar la variabilidad intersujeto e intra-sujeto dentro del mismo diseño experimental.
6. Definir un camino de escalamiento del estudio para ampliar la muestra, la complejidad del protocolo y la transferencia a escenarios clínicos.
7. Establecer una línea metodológica útil para trabajo diagnóstico y seguimiento en parálisis facial.

## 7. Hipótesis

La actividad muscular facial registrada por sEMG presenta patrones temporales y espectrales reproducibles asociados a expresiones faciales específicas. Cuando la secuencia, la sincronización y el etiquetado se definen a partir del protocolo experimental de Unity, es posible obtener una base de datos robusta y potencialmente aplicable a la evaluación de pacientes con parálisis facial. La hipótesis clínica es que la asimetría entre lados faciales, la capacidad de mantenimiento del gesto y la duración y amplitud de la activación muscular constituyen marcadores útiles para diferenciar patrones normales y alterados.

## 8. Diseño del estudio

### 8.1. Tipo de diseño

Estudio observacional, experimental, longitudinal, controlado y de laboratorio, con diseño incremental. La primera etapa consiste en una validación del protocolo en sujetos sanos. La segunda etapa consiste en la ampliación de la muestra para análisis de robustez y generalización. La tercera etapa, en sentido más clínico, se orienta a la aplicación en población con parálisis facial o patrones de disfunción facial funcional.

### 8.2. Fase actual del proyecto

La etapa actual corresponde a un protocolo experimental funcional y reproducible, en el que cada sujeto ejecuta repeticiones fijas de expresiones faciales bajo un flujo temporal claramente definido por Unity. La secuencia base incluye:

1. reposo inicial;
2. inicio de la secuencia experimental;
3. periodo de reposo previo;
4. cuenta regresiva visual;
5. mantenimiento del gesto facial solicitado;
6. pausa o reposo posterior;
7. repetición del ciclo para cada gesto;
8. finalización de la secuencia.

La temporalidad del protocolo se estructura en fases concretas:

- Reposo
- Countdown
- Gesto
- Eventos estructurales (inicio y cierre de secuencia)

Los eventos están definidos en Unity y se usan como referencia para el etiquetado. Los eventos relevantes incluyen:

- `sequence_start`
- `rest_start`
- `gesture_show`
- `countdown_3`, `countdown_2`, `countdown_1`
- `hold_start`
- `sequence_end`

Este esquema es consistente con el pipeline actual del proyecto, en el que las señales se sincronizan con los eventos de Unity, se segmentan y se etiquetan a partir del protocolo, y la fase de countdown se excluye de la clasificación por no representar la ejecución del gesto per se.

### 8.3. Especificación de la recolección actual

En la fase actual, se propone que cada sujeto complete 10 sesiones experimentales, distribuidas en 2 jornadas de recolección (5 sesiones por jornada) o, si la logística lo permite, 2 bloques de trabajo separados por varios días. La intención es la repetición de la misma secuencia con una ventana de tiempo razonable para capturar variabilidad intra-sujeto sin generar fatiga excesiva.

El protocolo base considera:

- 10 sesiones por sujeto;
- 2 sesiones de captura en un marco temporal más amplio o 2 jornadas de recolección;
- repeticiones fijas por sesión;
- secuencia estándar de gestos y pausas;
- mismo flujo temporal en todas las sesiones para permitir análisis comparativo.

La decisión de normalizar 10 sesiones por sujeto es clave para estudiar estabilidad, fatiga, consistencia y variabilidad intra-sujeto. La recolección en 2 bloques facilita la logística y reduce la fatiga durante una sola sesión prolongada.

### 8.4. Extensión posterior a la fase actual

Una vez que se cuenten datos suficientes de la fase base, se proyecta una segunda etapa de robustecimiento experimental. En este punto se plantea ampliar la complejidad del protocolo y la diversidad de condiciones, manteniendo los principios fundamentales del diseño: sincronización, etiquetado basado en eventos de Unity, y análisis fisiológico con señales sEMG.

La ampliación puede incluir:

- aumento del número de participantes (10 a 20 sujetos como primera etapa de robustez; más si la muestra lo requiere);
- sesiones repetidas por sujeto en varios días o semanas;
- variación de duración de reposo, cuenta regresiva y mantenimiento del gesto;
- inclusión de gestos más naturales o menos artificiales;
- incorporación de tareas de seguimiento visual o de respuesta en VR;
- protocolos con condiciones de movimiento, inmovilidad y postura alterada;
- comparación entre sujetos sanos y pacientes con alteración facial.

Esto permite pasar de un marco de validación técnica a un marco de generalización y aplicabilidad clínica.

## 9. Población objetivo, muestra y criterios de elegibilidad

### 9.1. Población objetivo

Personas adultas sanas con capacidad funcional normal para la musculatura facial. El estudio inicial se centra en población general sin discapacidad facial manifiesta, con el objetivo de construir una base de referencia para comparaciones futuras.

### 9.2. Muestra recomendada

La etapa de validación inicial puede centrarse en un rango de 10 a 20 sujetos. Se recomienda:

- 10 sujetos como umbral mínimo para análisis de estabilidad intersujeto inicial;
- 20 sujetos como objetivo prudente para análisis más robusto y posible evaluación de generalización;
- una muestra mayor en una etapa clínica posterior para comparar con población de parálisis facial.

La muestra de 10 a 20 sujetos es razonable en un primer estudio de robustez, porque permite observar variabilidad interindividual sin perder consistencia metodológica.

### 9.3. Criterios de inclusión

- edad adulta;
- capacidad para participar voluntariamente;
- comprensión del procedimiento y firma de consentimiento informado;
- ausencia de lesión facial activa o irritación importante en la zona de colocación de electrodos;
- capacidad para ejecutar las expresiones faciales bajo indicación visual o con guía de Unity.

### 9.4. Criterios de exclusión

- lesión facial reciente o potencialmente interferente;
- enfermedad neuromuscular relevante que altere la función facial;
- dermatitis, irritación cutánea o ulceración en la región de los electrodos;
- dolor facial o tensión marcada;
- contraindicación para uso del visor VR o incomodidad con la experiencia inmersiva;
- signos de fatiga extrema o incomodidad significativa durante la sesión.

## 10. Equipamiento y sistemas de adquisición

### 10.1. Sistema de sEMG

- Dispositivo: FREEEMG.
- Canales actuales: 4.
- Ubicación: 2 canales en la región frontal y 2 canales en la región de los cachetes.
- Región frontal: músculos asociados a la actividad frontal y supraciliar.
- Región de los cachetes: músculo cigomático mayor / buccinador y zonas adyacentes relevantes para la expresión facial.

### 10.2. IMU

- Dispositivo: Baiobit.
- Ubicación: parte posterior de la correa del visor de VR, en la región de la nuca.
- Función: registrar movimiento de cabeza, inclinación postural e información de posible desplazamiento que pueda contaminar la señal muscular facial.

### 10.3. Realidad virtual

- Dispositivo: Meta Quest 3.
- Función: presentar la secuencia experimental, las indicaciones de gesto y la cuenta regresiva, manteniendo la experiencia controlada y repetible.
- Uso: los estímulos visuales de Unity se presentan mediante el visor, y la sincronización del sistema se registra junto con la señal sEMG.

### 10.4. Protocolo y software

- La temporalidad y la secuencia de la tarea se controlan desde Unity.
- Los eventos experimentales se registran con marcas temporales y se usan como referencia del protocolo.
- El sistema de software tiene la capacidad de mapear eventos de Unity a fases (Reposo / Countdown / Gesto), con validación posterior.

## 11. Procedimiento experimental

### 11.1. Preparación del participante

Antes de la sesión se realizará:

- instrucción del procedimiento y objetivo del estudio;
- verificación de la salud facial y de la piel;
- revisión de la tolerancia al uso del visor VR;
- firma del consentimiento informado;
- preparación de la zona facial y colocación de electrodos.

### 11.2. Colocación de electrodos y calibración

Se colocarán los electrodos siguiendo el esquema anatómico del protocolo actual:

- 2 canales en la región frontal;
- 2 canales en la región de los cachetes;
- validación de la calidad de la señal antes del inicio;
- control del contacto, impedancia y estabilidad;
- registro inicial de línea base en reposo.

Se registrará la posición y el lado del participante para asegurar reproducibilidad y comparación entre sesiones.

### 11.3. Secuencia experimental actual

La secuencia del protocolo sigue la lógica de la estructura establecida en Unity:

1. reposo inicial;
2. inicio de la secuencia con `sequence_start`;
3. reposo previo al gesto con `rest_start`;
4. presentación del gesto / indicación visual con `gesture_show`;
5. cuenta regresiva con `countdown_3`, `countdown_2`, `countdown_1`;
6. fase de ejecución del gesto con `hold_start`;
7. retorno a reposo y cierre de la repetición;
8. continuación hasta completar la serie experimental.

La fase de Cuenta regresiva se considera un periodo preparatorio y no forma parte de la etiqueta de entrenamiento del gesto, pero se registra y conserva para análisis temporal, sincronización y control del protocolo.

### 11.4. Repeticiones y frecuencia de sesiones

La fase actual del estudio considera:

- 10 sesiones por sujeto;
- 2 bloques de sesión o dos jornadas de recolección;
- secuencia fija en cada una de las sesiones;
- posibilidad de repetir la misma tarea en varias condiciones de fatiga, postura o entorno visual si se requiere en etapas siguientes.

La presencia de 10 sesiones por sujeto permite cuantificar la estabilidad del patrón muscular, la variabilidad observada entre repeticiones, la fatiga o adaptación del usuario, y la consistencia del etiquetado experimental.

## 12. Protocolo experimental en Unity y lógica temporal

El protocolo de Unity constituye la base del experimento y se toma como referencia de verdad para el etiquetado. La secuencia fija es una ventaja metodológica porque estabiliza las relaciones temporales y reduce la posibilidad de ambigüedad en la segmentación.

El esquema temporal del protocolo se define de la siguiente manera:

- Reposo inicial: fase de base, sin ejecución del gesto.
- Presentación del gesto: se informa al sujeto qué expresión debe realizar.
- Countdown: preparación y anticipación.
- Gesto principal: duración fija o controlada del mantenimiento del gesto.
- Reposo post-gesto: recuperación and return to baseline.

El pipeline del proyecto establece que:

- todos los eventos de Unity tienen marcas temporales;
- la señal EMG se sincroniza con la secuencia de Unity;
- la etiqueta de cada segmento se deriva del evento temporal;
- la interpretación en base a la señal no reemplaza la verdad del protocolo experimental.

## 13. Descripción del flujo de análisis de datos

### 13.1. Adquisición y sincronización

- Se registran las señales EMG, los eventos de Unity y el IMU.
- Se validan marcas temporales y alineación de relojes.
- Se controlan huecos, solapamiento y consistencia de la secuencia.

### 13.2. Etiquetado

- Los eventos de Unity generan las fases del protocolo.
- La fase de `hold_start` define el gesto principal.
- La fase de `rest_start` define la clase de reposo.
- La fase de countdown se registra pero no se usa como clase de entrenamiento.

### 13.3. Preprocesamiento

- remoción de DC;
- filtro notch y pasa-banda adecuados;
- validación de la señal por canal y por ventana;
- recorte y normalización de la señal según los límites del protocolo.

### 13.4. Segmentación

La segmentación se realiza a partir de los intervalos del protocolo y se valida según duración, calidad de la señal, ausencia de ruido severo y contaminación por actividad no asociada al gesto.

### 13.5. Modelado y evaluación

- comparación de modelos de clasificación;
- validación cruzada intersujeto y/o leave-one-subject-out;
- evaluación por ventana y por segmento completo;
- métricas principales: macro-F1, balanced accuracy, accuracy y comparaciones por clase.

## 14. Fase de robustecimiento posterior: de 10 a 20 sujetos y condiciones ampliadas

Una vez consolidada la fase base, se proyecta una etapa de robustecimiento con los siguientes fines:

1. aumentar la heterogeneidad en la muestra;
2. analizar la variabilidad intra-sujeto y su evolución en varias sesiones;
3. evaluar si el protocolo fija mantiene su validez bajo cambios de tiempo, contexto y fatiga;
4. estudiar si existe una mejora real con incorporación de más canales y otras condiciones experimentales;
5. preparar la transición hacia población clínica con parálisis facial.

### 14.1. Condiciones posteriores sugeridas

Se propone ampliar el protocolo a condiciones como:

- diversas duraciones de gesto y reposo;
- diferentes órdenes de presentación de las expresiones;
- sesiones con y sin estímulo visual inmersivo;
- tareas realizadas en pantalla plana vs VR;
- condiciones de movimiento de cabeza o postura variable;
- condiciones con mayor fatiga para evaluar estabilidad.

### 14.2. Modificación del protocolo de Unity

El protocolo de Unity se volverá más robusto al incorporar:

- randomización de bloques de gestos;
- combinación de gestos espontáneos y guiados;
- duración variable de reposo y countown;
- protocolos de adaptación al participante;
- secuencias con mayor número de repeticiones y condiciones intermedias;
- registro de factores contextuales relevantes (fatiga, atención, tiempo de sesión, nivel de agotamiento).

Esto es importante porque un protocolo totalmente fijo es excelente para validación inicial, pero para estudios de generalización y para investigación clínica suele requerirse cierta variabilidad controlada.

### 14.3. Uso de VR y Meta Quest 3

El visor Meta Quest 3 puede ser útil para:

- controlar la atención del participante;
- ofrecer una experiencia inmersiva y repetible;
- presentar estímulos en una escena más natural o más demandante;
- crear escenarios con feedback visual y dinámicos.

Sin embargo, el uso de VR también introduce variables que deben controlarse:

- movimiento de cabeza;
- desplazamiento postural;
- fatiga visual;
- efecto de la inmersión sobre la expresión facial.

Por ello, se recomienda mantener el soporte de IMU y revisar si la posición del visor influye en la activación facial, y en etapas posteriores compararlo con una condición de visualización en pantalla convencional.

### 14.4. Aumento de canales para etapas posteriores

La fase actual con 4 canales es válida para validación inicial. Sin embargo, para futuras fases de investigación clínica, especialmente en parálisis facial, se recomienda:

- aumentar a 8 o más canales;
- añadir canales en zonas más periféricas de la musculatura facial;
- incorporar cobertura bilateral y simetría comparativa;
- incluir canales en la región periocular, orbicular, zygomaticus, levator labii, depressor anguli oris o buccinator, según el objetivo clínico.

Esto permite capturar mejor la asimetría y la diferencia funcional entre hemicaras, algo central en parálisis facial.

## 15. Proyección clínica y enfoque en parálisis facial

El estudio tiene una proyección claramente relevante para la evaluación de la parálisis facial. En un marco clínico, la información que se busca no es solo clasificar una expresión, sino detectar diferencias funcionales entre el lado sano y el afectado, cuantificar la asimetría y eventualmente seguir la evolución del tratamiento o la rehabilitación.

En esta línea, la fase clínica puede incluir:

- comparación entre participantes sanos y pacientes con parálisis facial;
- análisis de asimetría bilateral;
- comparación entre lado sano y lado afectado;
- medición de latencia, amplitud y duración de la contracción;
- seguimiento longitudinal antes y después de rehabilitación.

En la práctica clínica, el prototipo puede evolucionar hacia un sistema de apoyo diagnóstico o de monitoreo terapéutico, donde se evalúa si la actividad muscular facial de un paciente se aproxima a patrones sanos o muestra patrones persistentes de debilidad, asimetría o incoordinación.

## 16. Cronograma propuesto por semanas

Se propone un cronograma de 12 semanas para la fase completa de validación, robustecimiento y preparación clínica.

### Semana 1: preparación del protocolo y validación metodológica

Objetivo:
- definir la secuencia final de Unity;
- validar la colocación de electrodos y la sincronización;
- confirmar la calidad del flujo de adquisición.

Resultado esperado:
- protocolo experimental validado;
- señal EMG estable y sincronizada;
- primera versión del pipeline de etiquetado reproducible.

### Semana 2: pilotaje inicial

Objetivo:
- probar la metodología con 2 a 3 sujetos sanos;
- ajustar tiempos, margen de reposo y mantenimiento del gesto;
- revisar la fatiga, dificultad del visor y comodidad del participante.

Resultado esperado:
- versión refinada de la secuencia experimental;
- corrección de tiempos o ajustes en la colocación de electrodos;
- datos pilotaje con suficiente calidad para análisis inicial.

### Semana 3-4: primera etapa de recolección

Objetivo:
- reclutar 3 a 5 sujetos;
- completar 5 a 10 sesiones por sujeto en dos bloques de recolección;
- verificar estabilidad intra-sujeto.

Resultado esperado:
- conjunto de datos inicial con suficientes repeticiones por sujeto;
- evidencia de consistencia del protocolo;
- primeras métricas de rendimiento y calidad de señal.

### Semana 5-6: expansión intermedia

Objetivo:
- aumentar la muestra a 6 a 10 sujetos;
- evaluar el impacto de la variabilidad intersujeto;
- verificar si el protocolo fijo mantiene su validez de forma consistente.

Resultado esperado:
- base de datos con variabilidad moderada, útil para análisis exploratorio;
- primeras comparaciones por sujeto;
- identificación de factores que degradan la señal o el etiquetado.

### Semana 7-8: robustecimiento y análisis de generalización

Objetivo:
- llegar a 10 a 15 sujetos;
- comparar desempeño por sesión, por sujeto y por gesto;
- evaluar el efecto del reposo, la fatiga y la variación temporal.

Resultado esperado:
- robustez inicial del modelo y del protocolo;
- análisis estadístico de estabilidad y variabilidad;
- ajuste de estrategia de segmentación o de modelado.

### Semana 9-10: estudio de condiciones extendidas

Objetivo:
- introducir cambios controlados en la duración del gesto o en la secuencia visual;
- probar la experiencia VR vs pantalla plana;
- incluir un bloque con mayor complejidad del protocolo.

Resultado esperado:
- evidencia sobre qué condiciones influyen en los patrones musculares;
- mayor comprensión del papel de la VR y del IMU;
- decisión técnica sobre si se requiere mayor número de canales.

### Semana 11-12: preparación para aplicación clínica y cierre del estudio

Objetivo:
- consolidar el análisis final;
- definir la estrategia para pasar a pacientes con parálisis facial;
- preparar las métricas clínicas relevantes y la comparación con referencia.

Resultado esperado:
- protocolo final y resultados del estudio base;
- evaluación de capacidad diagnóstica inicial;
- propuesta formal para fase clínica con población afectada.

## 17. Resultados esperados y criterios de éxito

Se consideran resultados exitosos los siguientes:

- sincronización estable entre eventos de Unity y señal EMG;
- calidad de señal aceptable en el conjunto de sesiones;
- segmentación reproducible y concordante con el protocolo experimental;
- clasificación con desempeño superior al azar y consistente entre sujetos;
- evidencia de estabilidad a lo largo de las 10 sesiones por sujeto;
- identificación de factores que explican la variabilidad entre sujetos y entre sesiones.

Para la proyección clínica, el criterio de éxito es que el sistema permita distinguir patrones normales de actividad facial de patrones con asimetría or disfunción funcional, especialmente en casos de parálisis facial.

## 18. Consideraciones éticas y seguridad

- la participación será voluntaria;
- el protocolo es de bajo riesgo;
- el estudio no pretende provocar dolor ni daño permanente;
- la revisión cutánea y la supervisión del participante se realizarán en cada sesión;
- el uso de VR tendrá límites y controles para evitar incomodidad comparativa;
- los datos personales serán anonimizados y protegidos;
- el consentimiento informado incluirá claramente la finalidad, duración, riesgos y beneficios del estudio.

## 19. Conclusión

Este protocolo se presenta como un diseño integral que parte de una fase de validación técnica y experimental con población sana, basada en un protocolo fijo de Unity y en la adquisición de señales sEMG faciales con 4 canales, y se proyecta hacia una segunda etapa de robustecimiento con mayor número de sujetos, mayor complejidad del protocolo y expansión del sistema hacia escenarios más clínicos. La lógica metodológica del estudio se apoya en la sincronización, el etiquetado basado en protocolo y la reproducibilidad del procedimiento, lo cual resulta esencial para dar un paso posterior hacia la evaluación de parálisis facial y otras alteraciones de la actividad muscular facial.

El estudio actual no solo busca reconocer expresiones faciales, sino preparar una base sólida para la detección de alteraciones funcionales faciales, con la posibilidad de evolucionar hacia un enfoque de apoyo diagnóstico, seguimiento terapéutico y cuantificación de asimetría bilateral en población clínica.

## 20. Plantilla final para el comité de ética

Antes del envío formal, completar la siguiente información institucional:

- nombre completo del investigador responsable;
- nombre de la institución y del comité de ética receptor;
- número de versión del protocolo;
- fecha de aprobación y revisión;
- número estimado de participantes;
- distribución temporal de las sesiones y duración por participante;
- formulario definitivo de consentimiento informado;
- mecanismo de protección de datos y confidencialidad;
- línea de continuidad del estudio para la fase clínica en parálisis facial.
