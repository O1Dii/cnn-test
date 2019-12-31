import sys

import keras
import numpy as np
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout


def normalize_array(arr, amount=None):
    if not amount:
        amount = arr.shape[0]

    arr = arr.reshape(amount, 28, 28, 1)
    arr = arr.astype('float32')
    arr /= 255

    return arr


def load_custom_image(name='sample.png'):
    image = Image.open(name)
    initial_array = np.array(image)
    array = initial_array[:, :, 2]
    array = normalize_array(array, 1)

    return array


def prepare_data():
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = normalize_array(x_train)
    x_test = normalize_array(x_test)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def create_model():
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(32, input_shape=(70000, 10), activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    return model


def train_model(model, x_train, y_train, batch_size=128, epochs=1):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=3)])


app = QtWidgets.QApplication(sys.argv)
model = create_model()
(x_train, y_train), (x_test, y_test) = prepare_data()
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
# train_model(model, x, y, epochs=2)
model.load_weights('brain.h5')
image = load_custom_image()
init_prediction = model.predict(image)
print(init_prediction, np.argmax(init_prediction))
# model.save_weights('brain.h5')


class Window(QtWidgets.QWidget):
    window_height = 560
    window_width = 560

    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, self.window_height, self.window_width)
        self.setMinimumHeight(self.window_height)
        self.setMinimumWidth(self.window_width)
        self.points_set = set()
        self.pressed = False

        self.array = np.zeros((28, 28), dtype='int32')

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)

        qp.fillRect(0, 0, self.window_height, self.window_width, QColor(255, 255, 255))

        for point in self.points_set:
            qp.fillRect(point[0], point[1], 28, 28, QColor(0, 0, 0))

        qp.end()

    def mousePressEvent(self, QMouseEvent):
        self.pressed = True

    def mouseReleaseEvent(self, QMouseEvent):
        self.pressed = False

    def mouseMoveEvent(self, QMouseEvent):
        if self.pressed:
            x = QMouseEvent.x()
            y = QMouseEvent.y()
            self.array[y // 28, x // 28] = 255
            self.points_set.add(((x // 28) * 28, (y // 28) * 28))
            self.update()

    def get_array(self):
        return self.array

    def clear(self):
        self.array = np.zeros((28, 28), dtype='int32')
        self.points_set.clear()
        self.update()


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QHBoxLayout()

        self.result_layout = QVBoxLayout()
        self.label = QtWidgets.QLabel()
        self.canvas = Window()
        self.button = QtWidgets.QPushButton('Calculate')
        self.clear_button = QtWidgets.QPushButton('Clear')

        self.button.clicked.connect(self.button_click)
        self.clear_button.clicked.connect(self.clear_button_click)

        self.result_layout.addWidget(self.button)
        self.result_layout.addWidget(self.clear_button)
        self.result_layout.addWidget(self.label)

        self.layout.addWidget(self.canvas)
        self.layout.addLayout(self.result_layout)

        self.setLayout(self.layout)

    def button_click(self):
        image_arr = self.canvas.get_array()
        data = normalize_array(image_arr, 1)
        prediction = model.predict(data)
        print(prediction, np.argmax(prediction))
        self.label.setText(str(np.argmax(prediction)))

    def clear_button_click(self):
        self.canvas.clear()


window = MainWindow()
window.show()

sys.exit(app.exec_())
