import sys, os
sys.path.append(os.pardir)

import numpy as np
from DeepLearning_common.Layers import *
from DeepLearning_common.NeuralNetwork import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:

    def __init__(self, input_size_, hidden_size_, output_size_, weight_init_std_=0.01):
        # 重みの初期化
        self.params = {}
        self.params["W1"] = weight_init_std_ * np.random.randn(input_size_, hidden_size_)
        self.params["b1"] = np.zeros(hidden_size_)
        self.params["W2"] = weight_init_std_ * np.random.randn(hidden_size_, output_size_)
        self.params["b2"] = np.zeros(output_size_)

        # レイヤーの生成
        self.layers = OrderedDict()
        self.layers["Affine1"] = AffineLayer(self.params["W1"], self.params["b1"])
        self.layers["ReLU"] = ReLULayer()
        self.layers["Affine2"] = AffineLayer(self.params["W2"], self.params["b2"])
        self.last_layer = SoftmaxWithLossLayer()


    def predict(self, x_):
        for layer in self.layers.values():
            x_ = layer.forward(x_)

        return x_


    def loss(self, x_, t_):
        """

        :param x_: 入力データ
        :param t_: 教師データ（正解）
        :return: 最終層まで計算後の結果
        """

        y = self.predict(x_)
        return self.last_layer.forward(y, t_)


    def accuracy(self, x_, t_):
        y = self.predict(x_)
        y = np.argmax(y, axis=1)
        if t_.ndim != 1:
            t_ = np.argmax(t_, axis=1)

        accuracy = np.sum(y == t_) / float(x_.shape[0])
        return accuracy


    def numerical_gradient(self, x_, t_):
        """

        :param x_: 入力データ
        :param t_: 教師データ（正解）
        :return:
        """
        loss_W = lambda W: self.loss(x_, t_)

        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads


    def gradient(self, x_, t_):
        # forward
        self.loss(x_, t_)

        # backward
        delta_out = 1
        delta_out = self.last_layer.backward(delta_out)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            delta_out = layer.backward(delta_out)

        # 設定
        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db

        return grads

