import sys
import os

# Füge den aktuellen Ordner zum Python-Suchpfad hinzu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Nun kannst du 'get_data' importieren
from get_data import load_data

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# Baseline Modell (TF-IDF + SVM)
# --------------------------------------------------

def train_baseline_model(df, test_size=0.2, val_size=0.1):
    # Schritt 1: Train-Test-Split
    X = df["text"]
    y = df["type"]  # Hier z.B. 'type' als Zielvariable, kann aber auch 'queue' oder 'priority' sein
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )
    
    # Schritt 2: TF-IDF Vektorisierung
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5)  # Unigramme + Bigramme
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Schritt 3: SVM Modell
    svm_model = SVC(kernel='linear', random_state=42)  # lineares SVM
    svm_model.fit(X_train_tfidf, y_train)

    # Schritt 4: Vorhersagen
    y_pred = svm_model.predict(X_test_tfidf)

    # Schritt 5: Evaluierung
    print("SVM Baseline Evaluation - Classification Report:")
    # Genaueren Report (mehr Dezimalstellen) und Konfusionsmatrix ausgeben
    print(classification_report(y_test, y_pred, digits=4))
    
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    # Konfusionsmatrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    # Plot und speichern
    try:
        labels = np.unique(np.concatenate((y_test, y_pred)))
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(labels)),
            yticks=np.arange(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
            ylabel='True label',
            xlabel='Predicted label',
            title='Confusion Matrix'
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'confusion_matrix.png')
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved confusion matrix to {out_path}")
        # Save confusion matrix as CSV for easier parsing
        try:
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'confusion_matrix.csv')
            np.savetxt(csv_path, cm, fmt='%d', delimiter=',')
            print(f"Saved confusion matrix CSV to {csv_path}")
        except Exception as e:
            print("Could not save confusion matrix CSV:", e)

        # Write a short textual summary of main confusions
        try:
            summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'confusion_summary.txt')
            # Find off-diagonal maximum confusions per true class
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write('Confusion matrix summary\n')
                f.write('========================\n')
                labels_list = list(labels)
                for i, true_label in enumerate(labels_list):
                    row = cm[i].copy()
                    row[i] = -1  # ignore diagonal
                    max_idx = int(np.argmax(row))
                    max_val = int(cm[i, max_idx])
                    if max_val > 0:
                        f.write(f"True {true_label} -> most confused with Pred {labels_list[max_idx]}: {max_val} instances\n")
                    else:
                        f.write(f"True {true_label} -> no major confusions\n")
            print(f"Saved confusion summary to {summary_path}")
        except Exception as e:
            print("Could not write confusion summary:", e)
    except Exception as e:
        print("Could not plot confusion matrix:", e)

    return svm_model, tfidf_vectorizer


# Importiere die Funktion, falls sie aus einer anderen Datei kommt
from get_data import load_raw_dataframe
from get_data import create_text_features
from get_data import encode_labels


# Deine andere Logik für das Baseline-Modell
def compare_models():
    # Lade die Rohdaten und bereite die Textspalten vor (create_text_features sorgt dafür)
    df = load_raw_dataframe()
    df = create_text_features(df)  # Text-Spalte hinzufügen

    # Kodierung der Labels
    df, label_encoders = encode_labels(df)

    # Weiter mit dem Baseline-Modell (SVM mit TF-IDF)
    print("\nTraining SVM Baseline Model...")
    svm_model, tfidf_vectorizer = train_baseline_model(df)
    
    # Restlicher Code zum Trainieren des BERT-Modells
    # ...

if __name__ == "__main__":
    compare_models()

    
# --------------------------------------------------
# Ausführen der gesamten Pipeline
# --------------------------------------------------

if __name__ == "__main__":
    compare_models()
