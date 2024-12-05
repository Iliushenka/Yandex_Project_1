from random import uniform
from ActivationFunction import ActivationFunc
import sys


class Matrix:
    def __init__(self, row, column):
        self.matrix = [[0 for __ in range(column)] for _ in range(row)]
        self.row = row
        self.column = column

    def __str__(self):
        text = ["━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        for row in range(self.row):
            text.append(f"{row}: {self.get(row, 0)}")
        text += ["━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        return '\n'.join(text)

    def __repr__(self):
        text = ["━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        for row in range(self.row):
            text.append(f"{row}: {self.get(row, 0)}")
        text += ["━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        return '\n'.join(text)

    def rand(self):
        for row in range(self.row):
            for column in range(self.column):
                self.set(uniform(0, 1), row, column)
        return self

    def set_bias(self):
        for row in range(self.row):
            for column in range(self.column):
                self.set(1, row, column)
        return self

    def multi(self, weight, bias, activation):
        if self.row != weight.row:
            sys.exit("Error! Wrong row in multiplicative matrix")
        matrix = Matrix(weight.column, 1)
        for column in range(weight.column):
            delta = 0
            for row in range(weight.row):
                delta += self.get(row, 0) * weight.get(row, column)
            delta += bias.get(column, 0)
            delta = round(activation.forward(delta), 9)
            matrix.set(delta, column, 0)
        return matrix

    def set(self, value, row, column):
        self.matrix[row][column] = value

    def get(self, row, column):
        return self.matrix[row][column]

    def max_element(self):
        max_value, max_index = -100000, 0
        this_index = 0
        for row in range(self.row):
            for column in range(self.column):
                value = self.matrix[row][column]
                if value > max_value:
                    max_value, max_index = value, this_index
                this_index += 1
        return max_value, max_index
