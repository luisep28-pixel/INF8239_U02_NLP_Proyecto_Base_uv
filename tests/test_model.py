from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent


def test_model_artifacts_exist():
    model_path = BASE_DIR / "models" / "text_classifier_nb.pkl"
    vectorizer_path = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

    assert model_path.exists(), f"El archivo {model_path} no fue encontrado. Ejecute el entrenamiento primero."
    assert vectorizer_path.exists(), f"El archivo {vectorizer_path} no fue encontrado. Ejecute el entrenamiento primero."


def test_model_inference():
    model_path = BASE_DIR / "models" / "text_classifier_nb.pkl"
    vectorizer_path = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

    assert model_path.exists(), "Debe entrenar el modelo antes de validar inferencias."

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    samples = [
        "Hey, are we still meeting for lunch today?",  # ham -> 0
        "WINNER!! You have won a free iPhone. Call now!"  # spam -> 1
    ]

    X_sample = vectorizer.transform(samples)
    predictions = model.predict(X_sample)

    assert len(predictions) == len(samples)
    assert predictions[0] == 0
    assert predictions[1] == 1