import numpy as np
from DeepLearning_common import NeuralNetwork as nn


class MultipleLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    # z = x*y
    # x微分はy, y微分はx
    def backward(self, delta_out):
        dx = delta_out * self.y
        dy = delta_out * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    # z = x+y
    # x微分、y微分ともに1であり、そのまま
    def backward(self, delta_out):
        return delta_out, delta_out


class ReLULayer:
    def __init__(self):
        self.non_negative_flag = None

    def forward(self, x):
        self.non_negative_flag = (x >= 0)
        out = x.copy()
        out[np.logical_not(self.non_negative_flag)] = 0

        return out

    # 負の領域は０、非負の領域はそのまま
    def backward(self, delta_out):
        dx = delta_out.copy()
        dx[np.logical_not(self.non_negative_flag)] = 0

        return dx


class SigmoidLayer:
    def __init__(self):
        self.out = None

    def forawrd(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out


    def backward(self, delta_out):
        dx = delta_out * (1.0 - self.out) * self.out
        return dx


class AffineLayer:
    def __init__(self, W, b):
        """

        :param W:
        :param b:
        """
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, delta_out):
        dx = np.dot(delta_out, self.W.T)
        self.dW = np.dot(self.x.T, delta_out)
        self.db = np.sum(delta_out, axis=0)

        return dx


class SoftmaxWithLossLayer:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None


    def forward(self, x, t):
        self.t = t
        self.y = nn.softmax(x)
        self.loss = nn.cross_entropy_error(self.y, self.t)

        return self.loss


    def backward(self, delta_out=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

