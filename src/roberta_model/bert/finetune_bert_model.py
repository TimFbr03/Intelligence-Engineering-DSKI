import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from get_data import load_raw_dataframe
from get_data import create_text_features
from get_data import encode_labels

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

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

def train_svm_for_head(X_train_tfidf, X_test_tfidf, y_train, y_test, head_name, label_encoders):
    """Train SVM for a single classification head"""
    
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
    print(f"\n{head_name.upper()} - Classification Report:")
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    print(classification_report(y_test, y_pred, digits=4))

    # Calculate macro F1
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"{head_name} Macro F1-Score:", macro_f1)

    return svm_model, report, macro_f1


def train_all_heads(df, test_size=0.2):
    """Train SVM models for all three heads: type, queue, priority"""
    
    X = df["text"]
    
    X_train, X_test = train_test_split(
        X, test_size=test_size, random_state=42, shuffle=True
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

    # Get labels for all heads
    y_train_type = df.loc[X_train.index, "type"]
    y_test_type = df.loc[X_test.index, "type"]
    
    y_train_queue = df.loc[X_train.index, "queue"]
    y_test_queue = df.loc[X_test.index, "queue"]
    
    y_train_priority = df.loc[X_train.index, "priority"]
    y_test_priority = df.loc[X_test.index, "priority"]

    results = {}
    
    # Train models for each head
    for head_name, y_train, y_test in [
        ("type", y_train_type, y_test_type),
        ("queue", y_train_queue, y_test_queue),
        ("priority", y_train_priority, y_test_priority),
    ]:
        model, report, macro_f1 = train_svm_for_head(
            X_train_tfidf, X_test_tfidf, y_train, y_test, head_name, None
        )
        results[head_name] = {
            "model": model,
            "report": report,
            "macro_f1": macro_f1,
        }

    return tfidf_vectorizer, results


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
            "num_type": len(label_encoders["type"]),
            "num_queue": len(label_encoders["queue"]),
            "num_priority": len(label_encoders["priority"]),
        })

        print("\nTraining SVM models for all heads...")
        tfidf_vectorizer, results = train_all_heads(df)

        # Create reverse label encoders for readable class names
        reverse_encoders = {
            head: {v: k for k, v in encoders.items()}
            for head, encoders in label_encoders.items()
        }

        # Log metrics for each head
        for head_name, result in results.items():
            report = result["report"]
            macro_f1 = result["macro_f1"]
            
            # Log macro F1
            mlflow.log_metric(f"{head_name}_macro_f1", macro_f1)
            
            # Log per-class metrics
            for class_id, stats in report.items():
                if class_id.isdigit():
                    class_name = reverse_encoders[head_name][int(class_id)]
                    mlflow.log_metric(f"{head_name}_recall/{class_name}", stats["recall"])
                    mlflow.log_metric(f"{head_name}_precision/{class_name}", stats["precision"])
                    mlflow.log_metric(f"{head_name}_f1/{class_name}", stats["f1-score"])
                    mlflow.log_metric(f"{head_name}_support/{class_name}", stats["support"])

        # Log models
        mlflow.sklearn.log_model(tfidf_vectorizer, "tfidf_vectorizer")
        for head_name, result in results.items():
            mlflow.sklearn.log_model(result["model"], f"model_{head_name}")

        print("\nMLflow tracking complete!")


if __name__ == "__main__":
    compare_models()