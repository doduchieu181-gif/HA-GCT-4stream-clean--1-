"""Metrics không phụ thuộc sklearn; macro F1 tính trên TOÀN BỘ lớp cấu hình."""

import numpy as np


def classification_metrics(labels, scores, num_classes):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if not len(labels) or scores.shape != (len(labels), num_classes):
        raise ValueError("Metrics cần labels và scores không rỗng, đúng kích thước")
    predictions = scores.argmax(axis=1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    tp = confusion.diagonal()
    support = confusion.sum(1)
    predicted = confusion.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(num_classes), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(num_classes), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(num_classes), where=(precision + recall) > 0)
    k = min(5, num_classes)
    topk = np.argsort(scores, axis=1)[:, -k:]
    summary = {"samples": int(len(labels)), "top1": float((labels == predictions).mean()),
               f"top{k}": float((topk == labels[:, None]).any(axis=1).mean()),
               "macro_f1": float(f1.mean()), "topk_k": k}
    per_class = [{"label": i, "support": int(support[i]), "precision": float(precision[i]),
                  "recall": float(recall[i]), "f1": float(f1[i])} for i in range(num_classes)]
    return summary, confusion, per_class
