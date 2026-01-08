import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from get_data import load_data
from get_data import load_raw_dataframe
from get_data import create_text_features
from get_data import encode_labels

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------
# Baseline Modell (TF-IDF + SVM)
# --------------------------------------------------

def train_optimized_model(df, test_size=0.2):
    # Split
    X = df["text"]
    y = df["type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )

    # Optimierter TF-IDF
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Optimierte SVM
    svm_model = SVC(
        kernel='linear',
        C=3.0,
        class_weight='balanced',
        random_state=42
    )

    svm_model.fit(X_train_tfidf, y_train)

    # Prediction
    y_pred = svm_model.predict(X_test_tfidf)

    # Evaluation
    print("\nOptimized SVM Evaluation - Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    f1 = f1_score(y_test, y_pred, average='macro')
    print("Macro F1-Score:", f1)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    return svm_model, tfidf_vectorizer


def compare_models():
    df = load_raw_dataframe()
    df = create_text_features(df)
    df, _ = encode_labels(df)

    print("\nTraining SVM model...")
    train_optimized_model(df)


if __name__ == "__main__":
    compare_models()
