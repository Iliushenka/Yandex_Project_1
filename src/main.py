import warnings
from random import uniform, shuffle
import numpy as np
import sys, gzip, pickle
import time


# warnings.filterwarnings('ignore', category=RuntimeWarning)


class Matrix:
    def __init__(self, row, column):
        self.matrix = [[0 for __ in range(column)] for _ in range(row)]
        self.row, self.column = row, column

    def __str__(self):
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


class ActivationFunc:
    def __init__(self, activation):
        self.activation = activation.lower()
        if self.activation not in ("relu", "sigmoid"):
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
            if value < 0 or value > 0:
                return 0.01
            return 1
        elif activation == "sigmoid":
            return value * (1 - value)


class Network:
    def __init__(self, layers: str, activation='ReLu', learn_rate=0.01, bias_status='on'):
        self.layers = list()
        self.errors = list()
        self.bias = list()
        self.activation = ActivationFunc(activation=activation)
        self.lr = learn_rate

        self.commit(layers)
        self.L = len(self.layers)
        self.gen_bias(bias_status)
        self.bias_status = bias_status.lower()

        self.answer = None

    def save(self, filename):
        output = list()
        for weight_index in range(1, self.L - 1, 2):
            weight = self.layers[weight_index]
            omega_output = list()
            for row in range(weight.row):
                delta_output = list()
                for column in range(weight.column):
                    text = str(weight.get(row, column))
                    delta_output.append(text)
                delta_output = ' '.join(delta_output)
                omega_output.append(delta_output)
            omega_output = '\n'.join(omega_output)
            output.append(omega_output)
        output2 = list()
        for bias_index in range(0, self.L, 2):
            neuron = self.bias[bias_index]
            omega_output = list()
            for row in range(neuron.row):
                text = str(neuron.get(row, 0))
                omega_output.append(text)
            omega_output = ' '.join(omega_output)
            output2.append(omega_output)
        output = '<>'.join(output)
        output2 = '<>'.join(output2)
        output = output + '/' + output2
        open(f'../resource/{filename}', 'w').write(output)

    def load(self, filename):
        try:
            file_text = open(f'../resource/{filename}', 'r').read()
        except:
            sys.exit('Not found file!')
        weights, bias = file_text.split('/')
        weights = weights.split('<>')
        bias = bias.split('<>')
        layer = 1
        for weight in weights:
            rows = weight.split('\n')
            for row, value_row in enumerate(rows):
                columns = value_row.split(' ')
                for column, value_column in enumerate(columns):
                    self.layers[layer].set(float(value_column), row, column)
            layer += 2
        layer = 0
        for bias_layer in bias:
            rows = bias_layer.split(' ')
            for row, value_row in enumerate(rows):
                    self.bias[layer].set(float(value_row), row, 0)
            layer += 2

    def commit(self, layers):
        layers = [int(n) for n in layers.strip().split(' ')]
        delta_layers = list()
        delta_errors = list()
        for layer_index in range(len(layers) - 1):
            this_layer_size = layers[layer_index]
            next_layer_size = layers[layer_index + 1]
            delta = Matrix(this_layer_size, 1)
            delta_layers.append(delta)
            delta_errors.append(delta)
            delta = Matrix(this_layer_size, next_layer_size)
            delta_layers.append(delta.rand())
            delta_errors.append("Skip")
        delta = Matrix(layers[-1], 1)
        delta_layers.append(delta)
        delta_errors.append(delta)

        self.layers = delta_layers
        self.errors = delta_errors

    def gen_bias(self, status):
        for index_layer in range(0, self.L - 2, 2):
            this_layer = self.layers[index_layer]
            delta = Matrix(this_layer.row, 1)
            if status.lower() == 'on':
                delta.set_bias()
            self.bias += [delta]
            self.bias += ['Skip']
        last_layer = self.layers[-1]
        delta = Matrix(last_layer.row, 1)
        if status.lower() == 'on':
            delta.set_bias()
        self.bias += [delta]

    def update_lr(self, scalar):
        self.lr *= scalar

    def set_image(self, image, answer: int):
        index_image = 0
        for y_pixel in image:
            for x_pixel in y_pixel:
                self.layers[0].set(int(x_pixel), index_image, 0)
                index_image += 1
        self.answer = answer

    def forward(self):
        for n in range(0, self.L - 1, 2):
            delta = self.layers[n].multi(self.layers[n + 1], self.bias[n + 2], self.activation)
            self.layers[n + 2] = delta

        # guess_max, guess_index = self.layers[-1].max_element()
        # print(self.layers[-1], f"Answer: {self.answer}, Guess: {guess_index}", sep="")

    def backprop(self):
        for neuron in range(self.layers[-1].row):
            step = int(1 if neuron == self.answer else 0)
            delta = step - self.layers[-1].get(neuron, 0)
            self.errors[-1].set(delta, neuron, 0)
        # print(self.errors[-1])
        func = self.activation.backprop
        for weight_index in range(self.L - 2, -1, -2):
            for row in range(self.layers[weight_index].row):
                error = 0
                for column in range(self.layers[weight_index].column):
                    delta_weight = (self.layers[weight_index - 1].get(row, 0) *
                                    func(self.layers[weight_index + 1].get(column, 0)) * self.errors[weight_index + 1].
                                    get(column, 0) * self.lr)
                    error += (self.errors[weight_index + 1].get(column, 0) *
                              self.layers[weight_index].get(row, column) *
                              func(self.layers[weight_index + 1].get(column, 0)))
                    self.layers[weight_index].set(self.layers[weight_index].get(row, column) +
                                                  delta_weight, row, column)
                    if self.bias_status == 'on':
                        delta_bias = (self.errors[weight_index + 1].get(column, 0) * func(self.layers[weight_index + 1].
                                                                                          get(column, 0)) * self.lr)
                        self.bias[weight_index - 1].set(self.bias[weight_index - 1].get(row, 0) + delta_bias, row, 0)
                self.errors[weight_index - 1].set(error, row, 0)


with gzip.open('../resource/mnist.pkl.gz', 'rb') as f:
    if sys.version_info < (3,):
        data_ai = pickle.load(f)
    else:
        data_ai = pickle.load(f, encoding='bytes')
    f.close()
    (x_train, y_train), (x_test, y_test) = data_ai

layers_data = f"{28 * 28} 20 10"
network = Network(layers_data, activation="ReLu", learn_rate=0.085, bias_status='on')
network.load('local_data1.csv')

epochs = 10
start, end = (0, 100)
for epoch in range(1, epochs + 1):
    time_start = time.time()
    error_epoch = 0
    start += 0
    data = [n for n in range(start, start + end)]
    shuffle(data)
    print(f"Epoch: {epoch} / {epochs}")
    for index in data:
        network.set_image(x_train[index], y_train[index])
        network.forward()
        network.backprop()
        result_max, result_index = network.layers[-1].max_element()
        if int(network.answer) != int(result_index):
            error_epoch += 1
    network.update_lr(0.5)
    time_end = time.time()
    time_calc = time_end - time_start
    print(f"Errors: {error_epoch} / {end}, Calculated time: {time_calc} sec.")
print('End epochs!')
network.save('local_data1.csv')
