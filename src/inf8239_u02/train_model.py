import os
import joblib
from pathlib import Path
from dotenv import load_dotenv
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from inf8239_u02.preprocess import load_and_preprocess_data

load_dotenv()

# Determinar la raíz del proyecto dinámicamente
BASE_DIR = Path(__file__).resolve().parent.parent.parent


def train_and_evaluate():
    print("Cargando y preprocesando el corpus...")
    dataset_path = BASE_DIR / "data" / "raw" / "dataset.csv"

    if not dataset_path.exists():
        print(f"Error: No se encontró el dataset en {dataset_path}.")
        print("Por favor, ejecute primero: uv run python scripts/download_data.py")
        return

    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data(dataset_path)

    print("Entrenando el modelo de clasificación (Naive Bayes)...")
    model = MultinomialNB()
    model.fit(X_train, y_train)

    print("Evaluando el modelo...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nExactitud (Accuracy): {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=['ham (0)', 'spam (1)']))

    # Crear carpeta models/ en la raíz si no existe
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Guardar ambos artefactos .pkl
    joblib.dump(model, models_dir / "text_classifier_nb.pkl")
    joblib.dump(vectorizer, models_dir / "tfidf_vectorizer.pkl")

    print(f"✓ Éxito: Artefactos guardados exitosamente en: {models_dir}")


if __name__ == "__main__":
    train_and_evaluate()