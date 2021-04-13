import numpy as np

def multiclass_accuracy(prediction, ground_truth):
    return np.sum(prediction == ground_truth) / prediction.shape[0]
