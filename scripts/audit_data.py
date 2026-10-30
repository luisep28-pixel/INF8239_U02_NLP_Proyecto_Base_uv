import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def audit_dataset():
    path = os.getenv("DATASET_PATH", "data/raw/dataset.csv")
    text_col = os.getenv("TEXT_COLUMN", "text")
    target_col = os.getenv("TARGET_COLUMN", "label")

    if not Path(path).exists():
        print(f"Error: No se encuentra el archivo en {path}. Ejecute download_data.py primero.")
        return

    df = pd.read_csv(path)
    print("--- AUDITORÍA DEL CORPUS (Fuente: UCI ML Repository) ---")
    print(f"Total de observaciones (filas): {len(df)}")
    print(f"Total de columnas: {len(df.columns)}")
    print(f"Valores nulos en columna '{text_col}': {df[text_col].isnull().sum()}")
    print(f"Registros duplicados: {df.duplicated().sum()}")

    print("\nDistribución de clases (Target):")
    print(df[target_col].value_counts())

    # Métricas de longitud de texto requeridas por la guía de auditoría
    lengths = df[text_col].astype(str).apply(len)
    print("\nLongitud de caracteres (p50 / p90 / p99):")
    print(f"p50 (Mediana) : {lengths.quantile(0.50):.0f} caracteres")
    print(f"p90          : {lengths.quantile(0.90):.0f} caracteres")
    print(f"p99          : {lengths.quantile(0.99):.0f} caracteres")


if __name__ == "__main__":
    audit_dataset()