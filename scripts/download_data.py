import os
import io
import zipfile
import urllib.request
import hashlib
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def download_and_prepare_data():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = raw_dir / "dataset.csv"

    # URL oficial directa del repositorio UCI (UCI Machine Learning Repository ID: 228)
    uci_zip_url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

    if not dataset_path.exists():
        print("Descargando corpus público desde UCI Machine Learning Repository...")

        # Petición HTTP a la URL de UCI
        req = urllib.request.Request(
            uci_zip_url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )

        with urllib.request.urlopen(req) as response:
            zip_buffer = io.BytesIO(response.read())

        # Extraer el archivo 'SMSSpamCollection' dentro del ZIP de UCI
        with zipfile.ZipFile(zip_buffer) as z:
            with z.open("SMSSpamCollection") as f:
                # El archivo original de UCI viene separado por tabulaciones (TSV) y sin encabezados
                df = pd.read_csv(f, sep='\t', names=['label', 'text'], encoding='utf-8')

        # Guardar en formato CSV estandarizado para el laboratorio
        df.to_csv(dataset_path, index=False)
        print(f"Dataset descargado de UCI y procesado exitosamente.")

    # Calcular hash SHA-256 para auditoría de integridad
    sha256_hash = hashlib.sha256()
    with open(dataset_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    print(f"Dataset localizado en: {dataset_path}")
    print(f"Hash SHA-256 (UCI dataset): {sha256_hash.hexdigest()}")


if __name__ == "__main__":
    download_and_prepare_data()