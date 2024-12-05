import sys
import numpy as np


class ActivationFunc:
    def __init__(self, activation):
        self.activation = activation.lower()
        if self.activation not in ("relu", "sigmoid"):
            print('Такого аргумента не существует для функции активации!')
            sys.exit("Error! Wrong activation function")

    def forward(self, value):
        activation = self.activation
        if activation == "relu":
            if value <= 0:
                return value * 0.01
            elif value > 1:
                return 1 + 0.01 * (value - 1)
            return value
        elif activation == "sigmoid":
            value = np.clip(value, -709.78, 709.78)
            return 1 / (1 + np.exp(-value))

    def backprop(self, value):
        activation = self.activation
        if activation == 'relu':
            if value < 0:
                return 0.01
            elif value > 0:
                return 0.01
            return 1
        elif activation == "sigmoid":
            return value * (1 - value)
