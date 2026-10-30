# Cierre Interpretativo: Evaluación de Corpus y Clasificador de Texto NLP

**Asignatura:** INF-8239 Ciencia de Datos II  
**Unidad:** 02 · Procesamiento de Lenguaje Natural, redes y visión  
**Estudiante:** Luis Eduardo Puello Ureña  
**Código de Laboratorio:** U02.LAB04 / U02.LAB05  

---

## 1. Métrica y Resultado Principal
- **Modelo:** Naive Bayes Multinomial con Vectorización TF-IDF (3,000 características principales).
- **Exactitud Obtenida (Accuracy):** ~98% en el conjunto de prueba de retención (20% test split).
- **Desempeño por Clase:**
  - `ham` (Mensajes legítimos): Precision ~0.98 | Recall ~1.00
  - `spam` (Mensajes no deseados): Precision ~0.99 | Recall ~0.89

---

## 2. Evidencia de Calidad y Procedencia
- **Origen:** Dataset *SMS Spam Collection* proveniente del repositorio oficial **UCI Machine Learning Repository** (ID: 228).
- **Integridad:** Validada mediante sumas de verificación SHA-256 e inspección automatizada.
- **Contrato de Datos:** Verificado mediante pruebas unitarias (`pytest`), confirmando 0% de valores nulos en la columna de texto y una distribución de clases estructurada (~86% `ham`, ~14% `spam`).

---

## 3. Riesgo o Sesgo Identificado
1. **Desbalance de Clases:** Al existir significativamente más mensajes `ham` que `spam`, la métrica *Accuracy* puede resultar excesivamente optimista. Se priorizó el análisis del *F1-Score* y el *Recall* en la clase `spam`.
2. **Contexto Temporal e Idioma:** El corpus está compuesto por mensajes en inglés de la era de la mensajería SMS básica (abreviaciones, modismos de los 2000s). No cubre lenguaje técnico en español ni patrones de spam modernos de aplicaciones como WhatsApp o Telegram.

---

## 4. Decisión de Aprobación o Rechazo
- **Decisión:** **APROBADO**.
- **Justificación:** El corpus cumple con los estándares metodológicos de pertinencia, integridad, licencia abierta y reproducibilidad. Es óptimo para validar canalizaciones de vectorización TF-IDF y clasificadores probabilísticos en entornos de laboratorio.

---

## 5. Limitación que Debe Comunicarse
- El modelo generado (`text_classifier_nb.pkl`) no debe desplegarse directamente en entornos de producción reales para el filtrado de correos o mensajes modernos sin antes realizar un reentrenamiento (*fine-tuning*) con un dataset actualizado que incluya expresiones en español e imágenes/enlaces maliciosos.

---

## 6. Siguiente Verificación
- Integración de técnicas de balanceo de clases (ej. ponderación de clases o remuestreo) y comparación contra modelos lineales de clasificación (`LogisticRegression` o `LinearSVC`).