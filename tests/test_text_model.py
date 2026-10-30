from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent


def test_text_model_exists_and_predicts():
    model_path = BASE_DIR / "models" / "text_model.joblib"
    assert model_path.exists(), "El archivo 'models/text_model.joblib' no existe. Ejecute scripts/train_text.py."

    model = joblib.load(model_path)

    # Probar predicciones
    samples = ["Free entry to win a prize!", "Hey, call me when you get home."]
    preds = model.predict(samples)

    assert len(preds) == 2
    assert preds[0] == "spam"
    assert preds[1] == "ham"