from random import shuffle
import sys, gzip, pickle
import time

from PyQt6.QtCore import Qt, QPointF, QRect
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QMainWindow, QVBoxLayout
from PyQt6.QtGui import QPainter, QColor, QPalette, QPixmap, QFont

from Network import Network
from Matrix import Matrix
from ActivationFunction import ActivationFunc


class Canvas(QWidget):
    def __init__(self, data_label):
        super().__init__()
        self.setFixedSize(280, 280)
        self.setAutoFillBackground(True)
        self.setPalette(QPalette(QColor("black")))
        self.path = []
        self.data = data_label

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setClipping(False)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor("white"))
        painter.setOpacity(1)

        for i in range(len(self.path)):
            painter.drawEllipse(QPointF(self.path[i]), 6, 6)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.path.append(event.pos())
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.path.append(event.pos())
            self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.button_clear = QPushButton('Очистить', self)
        self.button_clear.move(279, 0)
        self.button_clear.resize(100, 100)
        self.button_clear.clicked.connect(self.clicked_clear)

        self.result = QLabel()
        canvas_result = QPixmap(10, 10)
        canvas_result.fill(Qt.GlobalColor.blue)
        self.result.setPixmap(canvas_result)
        self.result.move(280, 60)

        self.button_result = QPushButton('Запуск', self)
        self.button_result.move(559, 0)
        self.button_result.resize(100, 100)
        self.button_result.clicked.connect(self.clicked_result)

        self.label0 = QLabel(self)
        self.label0.move(280, 100)
        self.label0.setText('0: ')
        self.label0.resize(1000, 40)

        self.label1 = QLabel(self)
        self.label1.move(280, 150)
        self.label1.setText('1: ')
        self.label1.resize(1000, 40)

        self.label2 = QLabel(self)
        self.label2.move(280, 200)
        self.label2.setText('2: ')
        self.label2.resize(1000, 40)

        self.label3 = QLabel(self)
        self.label3.move(280, 250)
        self.label3.setText('3: ')
        self.label3.resize(1000, 40)

        self.label4 = QLabel(self)
        self.label4.move(280, 300)
        self.label4.setText('4: ')
        self.label4.resize(1000, 40)

        self.label5 = QLabel(self)
        self.label5.move(280, 350)
        self.label5.setText('5: ')
        self.label5.resize(1000, 40)

        self.label6 = QLabel(self)
        self.label6.move(280, 400)
        self.label6.setText('6: ')
        self.label6.resize(1000, 40)

        self.label7 = QLabel(self)
        self.label7.move(280, 450)
        self.label7.setText('7: ')
        self.label7.resize(1000, 40)

        self.label8 = QLabel(self)
        self.label8.move(280, 500)
        self.label8.setText('8: ')
        self.label8.resize(1000, 40)

        self.label9 = QLabel(self)
        self.label9.move(280, 550)
        self.label9.setText('9: ')
        self.label9.resize(1000, 40)

        self.result = QLabel(self)
        self.result.move(10, 280)
        self.result.setText('Результат: ')
        self.result.resize(1000, 40)

        self.data = [self.label0, self.label1, self.label2, self.label3, self.label4,
                     self.label5, self.label6, self.label7, self.label8, self.label9, self.result]

        self.canvas = Canvas(self.data)

        self.setCentralWidget(self.canvas)
        self.setStyleSheet("QLabel {font: 30pt Comic Sans MS}")

        self.setWindowTitle("AI program")
        self.setGeometry(100, 100, 700, 650)

    def clicked_result(self):
        results, index_max = result(self.canvas.path)
        self.label0.setText(f"0: {results[0][0]}")
        self.label1.setText(f"1: {results[1][0]}")
        self.label2.setText(f"2: {results[2][0]}")
        self.label3.setText(f"3: {results[3][0]}")
        self.label4.setText(f"4: {results[4][0]}")
        self.label5.setText(f"5: {results[5][0]}")
        self.label6.setText(f"6: {results[6][0]}")
        self.label7.setText(f"7: {results[7][0]}")
        self.label8.setText(f"8: {results[8][0]}")
        self.label9.setText(f"9: {results[9][0]}")
        self.result.setText(f"Результат: {index_max}")

    def clicked_clear(self):
        self.canvas.path = list()
        self.canvas.update()


blur = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.1, 0.4, 0.1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.4, 0.6, 0.4, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.1, 0.4, 0.1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]


def result(path):
    image = [[0 for __ in range(28)] for _ in range(28)]
    result_image = [[0 for __ in range(28)] for _ in range(28)]
    for pos in path:
        pos_y = max(min(pos.y() // 10, 27), 0)
        pos_x = max(min(pos.x() // 10, 27), 0)
        image[pos_y][pos_x] = 255
    for column in range(28):
        for row in range(28):
            delta = 0
            for m in range(-4, 5):
                for n in range(-4, 5):
                    delta_m, delta_n = row + m, column + n
                    if delta_n < 0 or delta_m < 0 or delta_m > 27 or delta_n > 27:
                        value = 0
                    else:
                        value = image[delta_m][delta_n] * blur[m + 4][n + 4]
                    delta += value
            result_image[row][column] = min(delta, 255)
    image = result_image
    network.set_image(image, 0)
    network.forward()
    return network.layers[-1].matrix, network.layers[-1].max_element()[1]



if __name__ == "__main__":
    layers_data = f"{28 * 28} 20 10"
    network = Network(layers_data, activation="relu", learn_rate=0.0001, bias_status='on')
    network.load('weight', 'weight4.csv')
    network.load('bias', 'bias4.csv')


    print("Добро пожаловать! ИИ инициализирован")
    situation = input('Обучить? 1/0: ')

    if situation == '1':
        with gzip.open('../resource/mnist.pkl.gz', 'rb') as f:
            if sys.version_info < (3,):
                data_ai = pickle.load(f)
            else:
                data_ai = pickle.load(f, encoding='bytes')
            f.close()
            (x_train, y_train), (x_test, y_test) = data_ai
        print("База данных загружена!")

        epochs = 5
        start, end = (3000, 1000)
        for epoch in range(1, epochs + 1):
            time_start = time.time()
            error_epoch = 0
            learn_update = 0.4
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
            network.update_lr(learn_update)
            time_end = time.time()
            time_calc = time_end - time_start
            print(f"Errors: {error_epoch} / {end}, Calculated time: {time_calc} sec.")
        print('End epochs!')
        print('Обучение заверешно!')
        network.save('weight', 'weight4.csv')
        network.save('bias', 'bias4.csv')
    elif situation == '0':
        print("Программа запущена")
        qApp = QApplication(sys.argv)
        app = MainWindow()
        app.show()
        sys.exit(qApp.exec())
    else:
        print("Надо написать 1 или 0")
