# Predicción de Clientes Elegibles — Internet Hogar

Modelo de clasificación para identificar clientes de telefonía móvil con alta
probabilidad de conversión al servicio de Internet Hogar, optimizando el gasto
de campañas de marketing.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-189fdd?style=flat)
![LightGBM](https://img.shields.io/badge/LightGBM-latest-2980b9?style=flat)

---

## Contexto del problema

Una empresa de telecomunicaciones busca expandir su servicio de Internet Hogar
aprovechando su base instalada de clientes móviles. El proceso tradicional de
prospección es masivo e indiscriminado, generando alto desperdicio de recursos.

Este proyecto construye un modelo predictivo que segmenta automáticamente a los
clientes según su elegibilidad, basándose en antigüedad, comportamiento financiero
e historial de servicio.

**Objetivo de negocio:** reducir el costo por contacto dirigiendo los esfuerzos
únicamente a clientes con alta probabilidad de conversión.

---

## Dataset

| Característica | Detalle |
|---|---|
| Registros | 5,000 clientes |
| Variables originales | 29 (numéricas, categóricas, booleanas, fechas) |
| Variable objetivo | `elegible_final` — binaria, 12.28% clase positiva |
| Fuente | Dataset sintético con estructura real de telecomunicaciones |

---

## Metodología

### 1. EDA (Análisis Exploratorio)
- Análisis de distribuciones, outliers y desbalance de clases (~12% positivos)
- Eliminación de columnas con más del 40% de nulos
- Corrección de columnas con valores mixtos string/bool

### 2. Detección y corrección de Data Leakage
Se detectó que varias variables booleanas son **criterios derivados de la misma
regla que construye `elegible_final`**, produciendo métricas artificialmente perfectas.
Se identificó la regla exacta y se corrigió entrenando solo con las **7 variables
crudas** disponibles en producción.

### 3. Features seleccionadas

| Variable | Tipo |
|---|---|
| `antiguedad_meses` | Numérica |
| `deuda_actual` | Numérica |
| `retrasos_6m` | Numérica |
| `reclamos_totales` | Numérica |
| `usuario_mi_movistar_activo` | Booleana |
| `email_habilitado` | Booleana |
| `tuvo_internet_hogar` | Booleana |

### 4. Modelos entrenados

Todos con el mismo split train/test (70/30, stratified, random_state=42)
y compensación de clases.

| Modelo | Compensación |
|---|---|
| Regresión Logística | `class_weight='balanced'` |
| Árbol de Decisiones | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight` automático |
| LightGBM | `class_weight='balanced'` |

### 5. Selección del modelo — F-beta (β=2)

Se usó **F-beta con β=2** como métrica de negocio (no ROC AUC), priorizando
Recall el doble que Precision dado que perder un cliente elegible tiene mayor
costo que contactar a uno no elegible.

El umbral de clasificación fue **optimizado individualmente** para cada modelo.

### 6. Validación cruzada y análisis de errores
- Validación cruzada 5-fold estratificada: todos los modelos estables (std < 0.02)
- Análisis de perfil de FN, FP y TP para entender con qué clientes falla el modelo

---

## Resultados

| Modelo | Umbral | Precision | Recall | F1 | F2 | ROC AUC |
|---|---|---|---|---|---|---|
| **XGBoost** | **0.05** | **0.4396** | **0.9891** | **0.6087** | **0.7913** | **0.9224** |
| Árbol de Decisiones | 0.23 | 0.4358 | 0.9783 | 0.6030 | 0.7833 | 0.9206 |
| LightGBM | 0.08 | 0.4348 | 0.9783 | 0.6020 | 0.7826 | 0.9236 |
| Regresión Logística | 0.62 | 0.4410 | 0.9348 | 0.5993 | 0.7638 | 0.9268 |

### Impacto de negocio (set de prueba — 1,500 clientes)

- Clientes recomendados para contactar: **414**
- De esos, realmente elegibles: **182 (44% tasa de conversión)**
- Elegibles no detectados: **2**

---

## Visualizaciones incluidas

- Distribución del target y desbalance de clases
- Heatmap de correlaciones
- Coeficientes de Regresión Logística y feature importance de XGBoost y LightGBM
- Curvas Precision-Recall y ROC para los 4 modelos
- Búsqueda de umbral óptimo por F-beta (gráfico por modelo)
- Matrices de confusión: umbral 0.5 vs umbral óptimo
- Validación cruzada: F2 por fold y detección de sobreajuste
- Análisis de errores: perfil de FN, FP y TP

---

## Autor

- Eduardo Castro Quicaña

Estudiante de Ingeniería de Sistemas orientado a **Data Analyst**.

LinkedIn  
https://www.linkedin.com/in/castroeduard

GitHub  
https://github.com/duardnek

## Equipo

- Castro Quicaña, Eduardo
- Garcia Berrocal, Bryan Alexander
- Quispe Zevallos, Anthony David
- Zavaleta Zavaleta, Angela
