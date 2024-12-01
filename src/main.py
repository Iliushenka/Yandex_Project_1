from random import shuffle
import sys, gzip, pickle
import time

from Network import Network
from Matrix import Matrix
from ActivationFunction import ActivationFunc


with gzip.open('../resource/mnist.pkl.gz', 'rb') as f:
    if sys.version_info < (3,):
        data_ai = pickle.load(f)
    else:
        data_ai = pickle.load(f, encoding='bytes')
    f.close()
    (x_train, y_train), (x_test, y_test) = data_ai


layers_data = f"{28 * 28} 20 10"
network = Network(layers_data, activation="ReLu", learn_rate=0.085, bias_status='on')
network.load('weight', 'weight1.csv')
network.load('bias', 'bias1.csv')

epochs = 20
start, end = (0, 10)
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
network.save('weight', 'weight1.csv')
network.save('bias', 'bias1.csv')
