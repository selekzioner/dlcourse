import numpy as np


def softmax(predictions):
    predictions_copy = np.copy(predictions)
    if (predictions_copy.ndim == 1):
        predictions_copy -= np.max(predictions_copy)
        exp_pred = np.exp(predictions_copy)
        exp_sum = np.sum(exp_pred)
        probs = exp_pred / exp_sum
    else:
        predictions_copy = (predictions_copy.T - np.max(predictions_copy, axis = 1)).T
        exp_pred = np.exp(predictions_copy)
        exp_sum = np.sum(exp_pred, axis = 1)
        probs = (exp_pred.T / exp_sum).T
        
    return probs


def cross_entropy_loss(probs, target_index):
    if (probs.ndim == 1):
        loss = -np.log(probs[target_index])
    else:
        batch_size = probs.shape[0]
        loss_arr = -np.log(probs[range(batch_size), target_index])
        loss = np.sum(loss_arr) / batch_size
        
    return loss


def softmax_with_cross_entropy(predictions, target_index):
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs
    if (probs.ndim == 1):
        dprediction[target_index] -= 1
    else:
        batch_size = predictions.shape[0]
        dprediction[range(batch_size), target_index] -= 1
        dprediction /= batch_size
        
    return loss, dprediction


def l2_regularization(W, reg_strength):
    loss = reg_strength * np.sum(np.square(W))
    grad = 2 * reg_strength * W

    return loss, grad
    

def linear_softmax(X, W, target_index):
    predictions = np.dot(X, W)
    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dprediction)
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            
            loss_sum = 0
            num_batches = int(num_train / batch_size)
            for batch_i in range(num_batches):
                loss, grad = linear_softmax(X[batches_indices[batch_i]], self.W, y[batches_indices[batch_i]])
                reg_loss, reg_grad = l2_regularization(self.W, reg)
                loss += reg_loss
                grad += reg_grad
                
                self.W -= learning_rate * grad
                loss_sum += loss
                loss_history.append(loss)
            # end
            loss = loss_sum / num_batches
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        prediction = np.dot(X, self.W)
        y_pred = np.where(prediction == np.amax(prediction, axis = 1).reshape(np.amax(prediction, 1).shape[0], 1))[1]
        return y_pred
