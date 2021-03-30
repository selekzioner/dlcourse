import numpy as np

  
def l2_regularization(W, reg_strength):
    loss = reg_strength * np.sum(np.square(W))
    grad = 2 * reg_strength * W
    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    preds_copy = np.copy(preds)
    preds_copy = (preds_copy.T - np.max(preds_copy, axis = 1)).T
    exp_pred = np.exp(preds_copy)
    exp_sum = np.sum(exp_pred, axis = 1)
    probs = (exp_pred.T / exp_sum).T
    
    batch_size = preds.shape[0]
    loss = np.sum(-np.log(probs[range(batch_size), target_index])) / batch_size
    
    d_preds = probs
    d_preds[range(batch_size), target_index] -= 1
    d_preds /= batch_size
    return loss, d_preds


class Param:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        return X * (X > 0)

    def backward(self, d_out):
        d_result = (self.X > 0) * d_out
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.sum(d_out, axis = 0, keepdims = True)
        d_result = np.dot(d_out, self.W.value.T)
        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}
