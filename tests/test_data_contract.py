import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def test_dataset_exists():
    path = os.getenv("DATASET_PATH", "data/raw/dataset.csv")
    assert Path(path).exists(), f"El archivo de datos no existe en la ruta obligatoria: {path}"


def test_data_contract_structure_and_integrity():
    path = os.getenv("DATASET_PATH", "data/raw/dataset.csv")
    text_col = os.getenv("TEXT_COLUMN", "text")
    target_col = os.getenv("TARGET_COLUMN", "label")

    df = pd.read_csv(path)

    # 1. Validar presencia de columnas obligatorias del contrato
    assert text_col in df.columns, f"Falta la columna de texto obligatoria: {text_col}"
    assert target_col in df.columns, f"Falta la columna objetivo obligatoria: {target_col}"

    # 2. Validar que no existan valores nulos en la columna de texto principal
    assert df[text_col].isnull().sum() == 0, "Se encontraron registros con texto nulo"

    # 3. Validar que el dataset tenga un volumen de observaciones válido (> 0)
    assert len(df) > 0, "El dataset se encuentra vacío"

    # 4. Validar que las etiquetas esperadas sean válidas (ham / spam)
    valid_labels = {'ham', 'spam'}
    unique_labels = set(df[target_col].unique())
    assert unique_labels.issubset(valid_labels), f"Se encontraron etiquetas no reconocidas: {unique_labels}"