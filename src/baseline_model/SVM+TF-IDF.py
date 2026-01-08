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
import mlflow
import mlflow.sklearn


# --------------------------------------------------
# Configuration
# --------------------------------------------------
EXPERIMENT_NAME = ""  # Link einfügen

mlflow.set_experiment(EXPERIMENT_NAME)


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
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    print(classification_report(y_test, y_pred, digits=4))

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    f1 = f1_score(y_test, y_pred, average='macro')
    print("Macro F1-Score:", f1)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    return svm_model, tfidf_vectorizer, report, acc, f1


def compare_models():
    df = load_raw_dataframe()
    df = create_text_features(df)
    df, label_encoders = encode_labels(df)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "model": "SVM + TF-IDF",
            "kernel": "linear",
            "C": 3.0,
            "class_weight": "balanced",
            "ngram_range": "(1, 2)",
            "min_df": 3,
            "max_df": 0.9,
            "sublinear_tf": True,
            "test_size": 0.2,
            "num_type_classes": len(label_encoders["type"]),
        })

        print("\nTraining SVM model...")
        svm_model, tfidf_vectorizer, report, acc, f1 = train_optimized_model(df)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("macro_f1", f1)
        mlflow.log_metric("weighted_f1", report["weighted avg"]["f1-score"])

        # Log per-class metrics
        for class_name, stats in report.items():
            if class_name not in ["accuracy", "macro avg", "weighted avg"]:
                mlflow.log_metric(f"recall/{class_name}", stats["recall"])
                mlflow.log_metric(f"precision/{class_name}", stats["precision"])
                mlflow.log_metric(f"f1/{class_name}", stats["f1-score"])

        # Log models
        mlflow.sklearn.log_model(svm_model, "svm_model")
        mlflow.sklearn.log_model(tfidf_vectorizer, "tfidf_vectorizer")

        print("\nMLflow tracking complete!")


if __name__ == "__main__":
    compare_models()