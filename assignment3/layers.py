import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    loss = reg_strength * np.sum(np.square(W))
    grad = 2 * reg_strength * W
    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    preds_copy = np.copy(predictions)
    preds_copy = (preds_copy.T - np.max(preds_copy, axis = 1)).T
    exp_pred = np.exp(preds_copy)
    exp_sum = np.sum(exp_pred, axis = 1)
    probs = (exp_pred.T / exp_sum).T
    
    batch_size = predictions.shape[0]
    loss = np.sum(-np.log(probs[range(batch_size), target_index])) / batch_size
    
    dprediction = probs
    dprediction[range(batch_size), target_index] -= 1
    dprediction /= batch_size
    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X
        return X * (X > 0)

    def backward(self, d_out):
        # TODO copy from the previous assignment
        d_result = (self.X > 0) * d_out
        return d_result

    def params(self):
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
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )
        self.B = Param(np.zeros(out_channels))
        self.padding = padding
        self.X = None


    def forward(self, X):
        batch_size, height, width, channels = X.shape
        
        out_height = height - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1
        
        self.X = np.pad(X, ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant', constant_values = 0)
        W = self.W.value.reshape(self.filter_size * self.filter_size * self.in_channels, self.out_channels)
        
        out_shape = (batch_size, out_height, out_width, self.out_channels)
        output = np.zeros(out_shape)
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                h_end, w_end = y + self.filter_size, x + self.filter_size
                
                I = self.X[:, y:h_end, x:w_end, :].reshape(batch_size, -1)
                output[:, y, x, :] = np.dot(I, W)
                
        return output + self.B.value


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        dX = np.zeros_like(self.X)
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                h_end, w_end = y + self.filter_size, x + self.filter_size
                
                dX[:, y:h_end, x:w_end, :] += np.dot(d_out[:, y, x, :], self.W.value.reshape(
                    -1, self.out_channels).T).reshape(batch_size, -1)
                
                self.W.grad += np.dot(self.X[:, y:h_end, x:w_end, :].reshape(
                    batch_size, -1).T, d_out[:, y, x, :]).reshape(self.W.value.shape)
                
        self.B.grad = np.sum(d_out, axis = (0, 1, 2))
        return dX[:, self.padding : (height - self.padding), self.padding : (width - self.padding), :]

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.masks = {}

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        self.masks.clear()
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        output_shape = (batch_size, out_height, out_width, channels)
        output = np.zeros(output_shape)
        
        for y in range(out_height):
            for x in range(out_width):
                h_begin, w_begin = y * self.stride, x * self.stride
                h_end, w_end = h_begin + self.pool_size, w_begin + self.pool_size
                
                I = X[:, h_begin:h_end, w_begin:w_end, :]
                self.build_mask(x=I, pos=(y, x))
                output[:, y, x, :] = np.max(I, axis=(1, 2))
            
        return output
        

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        dX = np.zeros_like(self.X)
        
        for y in range(out_height):
            for x in range(out_width):
                h_begin, w_begin = y * self.stride, x * self.stride
                h_end, w_end = h_begin + self.pool_size, w_begin + self.pool_size
                
                dX[:, h_begin:h_end, w_begin:w_end, :] += d_out[:, y:y + 1, x:x + 1, :] * self.masks[(y, x)]   
        return dX
    
    def build_mask(self, x, pos):
        mask = np.zeros_like(x)
        n, h, w, c = x.shape
        x = x.reshape(n, h * w, c)
        idx = np.argmax(x, axis=1)

        n_idx, c_idx = np.indices((n, c))
        mask.reshape(n, h * w, c)[n_idx, idx, c_idx] = 1
        self.masks[pos] = mask

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
