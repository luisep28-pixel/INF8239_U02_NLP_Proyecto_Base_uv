import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(csv_path="data/raw/dataset.csv"):
    df = pd.read_csv(csv_path)

    # Limpieza básica de nulos por seguridad
    df = df.dropna(subset=['text', 'label'])

    # Mapeo de la etiqueta a variable numérica binaria (ham: 0, spam: 1)
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'],
        df['target'],
        test_size=0.2,
        random_state=42,
        stratify=df['target']
    )

    # Vectorización TF-IDF
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer