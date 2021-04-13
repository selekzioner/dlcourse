import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        self.conv_1 = ConvolutionalLayer(in_channels = 3, out_channels = 3, filter_size = conv1_channels, padding = 1)
        self.relu_1 = ReLULayer()
        self.pool_1 = MaxPoolingLayer(pool_size = 4, stride = 2)
        self.conv_2 = ConvolutionalLayer(in_channels = 3, out_channels = 3, filter_size = conv2_channels, padding = 1)
        self.relu_2 = ReLULayer()
        self.pool_2 = MaxPoolingLayer(pool_size = 4, stride = 2)
        self.flat = Flattener()
        self.fc = FullyConnectedLayer(n_input = 147, n_output = n_output_classes)
    
    def forward_pass(self, X):
        out = self.fc.forward(
            self.flat.forward(
            self.pool_2.forward(self.relu_2.forward(self.conv_2.forward(
            self.pool_1.forward(self.relu_1.forward(self.conv_1.forward(X)
                                                   )))))))
        return out
    
    def backward_pass(self, d_out):
        d_result = self.conv_1.backward(
            self.relu_1.backward(
            self.pool_1.backward(self.conv_2.backward(self.relu_2.backward(
            self.pool_2.backward(self.flat.backward(self.fc.backward(d_out)
                                                   )))))))
        return d_result
    
    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)
        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        out = self.forward_pass(X)
        loss, d_out = softmax_with_cross_entropy(out, y)
        self.backward_pass(d_out)
        
        return loss

    def predict(self, X):
        out = self.forward_pass(X)
        pred = np.argmax(out, axis = 1)
        return pred

    def params(self):
        result = {'W1': self.conv_1.params()['W'], 'B1': self.conv_1.params()['B'], 
                'W2': self.conv_2.params()['W'], 'B2': self.conv_2.params()['B'], 
                 'W3': self.fc.params()['W'], 'B3': self.fc.params()['B']}
        # TODO: Aggregate all the params from all the layers
        # which have parameters
        return result
