import streamlit as st
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "text_model.joblib"

st.set_page_config(page_title="Clasificador de Mensajes SMS", page_icon="✉️")

st.title("Clasificador de Texto SMS (Spam / Ham)")
st.caption("INF-8239 Ciencia de Datos II · U02.LAB05")

# Advertencia de alcance
st.warning(
    "⚠️ **Nota de alcance:** Este modelo fue entrenado con un corpus de SMS en inglés de la UCI ML Repository. "
    "Sus predicciones son probabilísticas y están condicionadas al vocabulario de dicho corpus."
)

if not MODEL_PATH.exists():
    st.error("Modelo no encontrado. Por favor ejecute `uv run python scripts/train_text.py` primero.")
else:
    model = joblib.load(MODEL_PATH)

    user_input = st.text_area("Ingrese el texto del mensaje a analizar:", "WINNER!! You have won a 1000 cash prize!")

    if st.button("Clasificar Mensaje"):
        if user_input.strip() == "":
            st.warning("Por favor ingrese algún texto.")
        else:
            pred = model.predict([user_input])[0]

            # Obtener probabilidades si el modelo lo soporta
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([user_input])[0]
                classes = model.classes_
                prob_dict = dict(zip(classes, probs))
                confidence = prob_dict[pred] * 100
            else:
                confidence = None

            if pred == "spam":
                st.error(f"🚨 **Resultado:** SPAM" + (f" (Confianza: {confidence:.2f}%)" if confidence else ""))
            else:
                st.success(f"✅ **Resultado:** HAM (Mensaje Legítimo)" + (
                    f" (Confianza: {confidence:.2f}%)" if confidence else ""))