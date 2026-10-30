# Informe de Resultados de Laboratorios: U02.LAB04 y U02.LAB05
**Asignatura:** INF-8239 Ciencia de Datos II  
**Estudiante:** Luis Eduardo Puello Ureña  
**Institución:** Universidad Autónoma de Santo Domingo (UASD)  
**Repositorio GitHub:** https://github.com/luisep28-pixel/INF8239_U02_NLP_Proyecto_Base_uv  

---

### 1. Resumen Ejecutivo y Procedencia de Datos
En la realización del laboratorio U02.LAB04 se localizó, evaluó y auditó el corpus público **"SMS Spam Collection Data Set"** alojado en el repositorio oficial de la **UCI Machine Learning Repository** (ID: 228). El conjunto de datos consta de 5,572 observaciones etiquetadas binariamente (`ham` para mensajes legítimos y `spam` para correo no deseado). Durante la auditoría automatizada se comprobó un 0% de registros nulos en la columna de texto y una distribución de clases de aproximadamente 86.6% `ham` y 13.4% `spam`.

### 2. Metodología de Modelado y Comparativa (U02.LAB05)
Para el desarrollo del laboratorio U02.LAB05 se construyó una arquitectura reproducible separando el preprocesamiento dentro de cada pipeline mediante vectorizaciones TF-IDF (3,000 características principales) y una partición estratificada con semilla 42 (80% entrenamiento / 20% prueba). Se entrenaron e inspeccionaron tres modelos comparables:
1. **Baseline (DummyClassifier):** Estrategia 'prior'. Obtuvo un F1 Macro de ~0.46, confirmando la necesidad de modelos supervisados más complejos.
2. **Complement Naive Bayes:** Diseñado para manejar el desbalance de clases, alcanzando un F1 Macro de ~0.94.
3. **Regresión Logística:** Obtuvo el desempeño superior con un **F1 Macro de ~0.95**, un Accuracy global de ~0.98, Precision de ~0.99 para Spam y un Recall de ~0.90 para la misma clase.

### 3. Inspección de Errores y Matriz de Confusión
El análisis de la matriz de confusión del modelo de Regresión Logística reveló que la clase con mayor dificultad fue `spam` en términos de *Recall*, registrándose un pequeño porcentaje de falsos negativos (mensajes de spam clasificados erróneamente como `ham`).

Al inspeccionar los 20 errores principales almacenados en `reports/error_analysis.csv`, los tipos de error más frecuentes se categorizaron en:
- **Ambigüedad y Texto Insuficiente:** Mensajes extremadamente cortas como *"OK"*, *"Yes"*, o códigos alfanuméricos ambiguos.
- **Dialecto y Modismos:** Expresiones informales en inglés escritas con jerga de mensajería de texto de los años 2000 (ej. *"txt"*, *"u"*, *"n"*, *"4"*), que no siempre acumulan peso significativo en la matriz TF-IDF.

### 4. Conclusión e Impacto en el Contexto
Aunque el modelo final de Regresión Logística presenta excelentes métricas globales, la principal limitación del dataset radica en que está compuesto únicamente por SMS antiguos en idioma inglés. En un escenario de producción real en nuestro contexto regional (República Dominicana), este modelo no sería capaz de clasificar spam en español ni detectar amenazas modernas en plataformas de mensajería instantánea (como phishing con enlaces maliciosos en WhatsApp).

**Decisión final:** Se **APRUEBA** la canalización como Baseline técnico interpretable y ejecutable localmente mediante la aplicación interactiva de Streamlit (`app/streamlit_app.py`), recomendando un reentrenamiento con corpora multilingües actualizados antes de cualquier despliegue comercial.