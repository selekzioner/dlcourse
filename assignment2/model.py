import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        self.fc1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.rl = ReLULayer()
        self.fc2 = FullyConnectedLayer(hidden_layer_size, n_output)        
        self.reg = reg

    def compute_loss_and_gradients(self, X, y):
        for param in self.params().values():
            param.grad = 0
        out = self.fc2.forward(self.rl.forward(self.fc1.forward(X)))
        loss, grad = softmax_with_cross_entropy(out, y)
        self.fc1.backward(self.rl.backward(self.fc2.backward(grad)))
        
        for param in self.params().values():
            reg_loss, reg_grad = l2_regularization(param.value, self.reg) 
            loss += reg_loss
            param.grad += reg_grad
        return loss

    def predict(self, X):
        out = self.fc2.forward(self.rl.forward(self.fc1.forward(X)))
        pred = np.argmax(out, axis = 1)
        return pred

    def params(self):
        return {'W1': self.fc1.params()['W'], 'B1': self.fc1.params()['B'], 
                'W2': self.fc2.params()['W'], 'B2': self.fc2.params()['B']}
