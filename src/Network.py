import sys

from ActivationFunction import ActivationFunc
from Matrix import Matrix


class Network:
    def __init__(self, layers: str, activation='ReLu', learn_rate=0.1, bias_status='on'):
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

    def save(self, type, filename):
        type = type.lower()
        filename = filename.lower()
        if type not in ('weight', 'bias'):
            sys.exit('Try "weights", "bias" for save')
        output = list()
        if type == 'weight':
            for weight_index in range(1, self.L - 1, 2):
                weight = self.layers[weight_index]
                delta_output = list()
                for row in range(weight.row):
                    alpha_output = list()
                    for column in range(weight.column):
                        text = str(weight.get(row, column))
                        alpha_output.append(text)
                    alpha_output = '_'.join(alpha_output)
                    delta_output.append(alpha_output)
                delta_output = ' '.join(delta_output)
                output.append(delta_output)
            output = '\n'.join(output)
        elif type == "bias":
            for bias_index in range(2, self.L, 2):
                bias = self.layers[bias_index]
                delta_output = list()
                for row in range(bias.row):
                        text = str(bias.get(row, 0))
                        delta_output.append(text)
                delta_output = ' '.join(delta_output)
                output.append(delta_output)
            output = '\n'.join(output)
        open(f'../resource/{filename}', 'w').write(output)
        print(f'Сохранено {type}')

    def load(self, type, filename):
        type = type.lower()
        filename = filename.lower()
        file_text = ''
        try:
            file_text = open(f'../resource/{filename}', 'r').read()
        except:
            sys.exit('Not found file!')
        if type not in ('weight', 'bias'):
            sys.exit('Try "weight", "bias" for load')
        try:
            if type == "weight":
                weights = file_text.split('\n')
                layer = 1
                for weight in weights:
                    rows = weight.split(' ')
                    for row, value_row in enumerate(rows):
                        columns = value_row.split('_')
                        for column, value_column in enumerate(columns):
                            self.layers[layer].set(float(value_column), row, column)
                    layer += 2
            elif type == "bias":
                weights = file_text.split('\n')
                layer = 2
                for weight in weights:
                    rows = weight.split(' ')
                    for row, value_row in enumerate(rows):
                        self.layers[layer].set(float(value_row), row, 0)
                    layer += 2
            print(f'Загружено {type}')
        except:
            sys.exit(f'Error in data with filename "{filename}"')


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
        # act = self.activation.backprop
        # for index in range(10):
        #     cost = 0
        #     if index == self.answer:
        #         cost = 1
        #     delta = cost - self.layers[-1].get(index, 0)
        #     self.errors[-1].set(delta, index, 0)
        # # print(self.errors[-1])
        # for m in range(self.layers[3].row):
        #     info = 0
        #     for n in range(self.layers[3].column):
        #         error = (self.layers[2].get(m, 0) * act(self.layers[4].get(n, 0)) * self.errors[4].get(n, 0)
        #                  * self.lr)
        #         info += self.errors[4].get(n, 0) * self.layers[3].get(m, n) * act(self.layers[4].get(n, 0))
        #         self.layers[3].set(self.layers[3].get(m, n) + error, m, n)
        #     self.errors[2].set(info, m, 0)
        # for m in range(self.layers[1].row):
        #     for n in range(self.layers[1].column):
        #         error = (self.layers[0].get(m, 0) * act(self.layers[2].get(n, 0)) * self.errors[2].get(n, 0)
        #                  * self.lr)
        #         self.layers[1].set(self.layers[1].get(m, n) + error, m, n)

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
