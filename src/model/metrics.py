import evaluate
import numpy as numpy
from sklearn.metrics import confusion_matrix

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    '''

    '''
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(predictions=predictions, reference=labels)
    f1 = f1_metric.compute(predictions=predictions, reference=labels, average="weighted")

    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"]
    }

def compute_confusion_matrix(logits, labels):
    preds = np.argmax(logits, axis=-1)
    return confusion_matrix(labels, preds)
