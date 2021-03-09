import numpy as np
import collections

def binary_classification_metrics(prediction, ground_truth):
    
    counts = collections.Counter(zip(prediction, ground_truth))
    
    precision = counts[1, 1] / (counts[1, 1] + counts[1, 0])
    recall = counts[1, 1] / (counts[1, 1] + counts[0, 1])
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (counts[1, 1] + counts[0, 1]) / (counts[1, 1] + counts[0, 1] + counts[1, 0] + counts[0, 0])
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    accuracy = np.where(prediction == ground_truth)[0].shape[0] / prediction.shape[0]
    return accuracy
