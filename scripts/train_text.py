import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import joblib
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def run_experiment():
    dataset_path = BASE_DIR / "data" / "raw" / "dataset.csv"
    if not dataset_path.exists():
        print(f"Error: No se encontró el dataset en {dataset_path}")
        return

    # 1. Cargar datos
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=['text', 'label'])

    # Mapeo a variable binaria
    X = df['text']
    y = df['label']

    # 2. Partición estratificada con semilla 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Definir Pipelines
    pipelines = {
        'Baseline (Dummy)': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english')),
            ('clf', DummyClassifier(strategy='prior'))
        ]),
        'Complement Naive Bayes': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english')),
            ('clf', ComplementNB())
        ]),
        'Regresión Logística': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english')),
            ('clf', LogisticRegression(random_state=42, max_iter=1000))
        ])
    }

    # 4. Entrenar y Evaluar
    metrics_list = []
    best_f1 = -1.0
    best_pipeline = None
    best_name = ""
    best_y_pred = None

    print("--- ENTRENAMIENTO DE PIPELINES DE CLASIFICACIÓN ---")
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        f1_macro = f1_score(y_test, y_pred, average='macro')
        print(f"• {name} - F1 Macro: {f1_macro:.4f}")

        report = classification_report(y_test, y_pred, output_dict=True)
        metrics_list.append({
            'Modelo': name,
            'F1_Macro': f1_macro,
            'Accuracy': report['accuracy'],
            'Precision_Ham': report['ham']['precision'],
            'Recall_Ham': report['ham']['recall'],
            'Precision_Spam': report['spam']['precision'],
            'Recall_Spam': report['spam']['recall']
        })

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_pipeline = pipe
            best_name = name
            best_y_pred = y_pred

    # 5. Guardar Métricas en CSV
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(reports_dir / "text_metrics.csv", index=False)
    print(f"\n[✓] Métricas guardadas en: {reports_dir / 'text_metrics.csv'}")

    # 6. Generar y Guardar Matriz de Confusión del Mejor Modelo
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, best_y_pred, labels=['ham', 'spam'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham', 'spam'])
    disp.plot(cmap='Blues', ax=ax)
    plt.title(f"Matriz de Confusión: {best_name}")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_text.png")
    plt.close()
    print(f"[✓] Gráfica de confusión guardada en: {reports_dir / 'confusion_text.png'}")

    # 7. Generar Análisis de Errores (Primeros errores clasificados)
    df_test = pd.DataFrame({'texto': X_test, 'real': y_test, 'predicho': best_y_pred})
    df_errors = df_test[df_test['real'] != df_test['predicho']].copy()

    # Categorización cualitativa inicial
    df_errors['categoria_error'] = 'Ambigüedad/Dialecto'
    df_errors.head(25).to_csv(reports_dir / "error_analysis.csv", index=False)
    print(f"[✓] Muestra de errores guardada en: {reports_dir / 'error_analysis.csv'}")

    # 8. Guardar el Mejor Modelo
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, models_dir / "text_model.joblib")
    print(f"[✓] Mejor modelo ({best_name}) guardado en: {models_dir / 'text_model.joblib'}")


if __name__ == "__main__":
    run_experiment()