import numpy as np


class SGD:
    def update(self, w, d_w, learning_rate):
        return w - d_w * learning_rate


class MomentumSGD:
    
    def __init__(self, momentum=0.9):
        self.momentum = 0.9
        self.velocity = 0
    
    def update(self, w, d_w, learning_rate):
        self.velocity = self.momentum * self.velocity - learning_rate * d_w 
        w += self.velocity
        return w
