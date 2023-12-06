import os
from PIL import ImageOps
import PIL.Image
import PIL
import numpy as np
from tkinter import *
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import utils
from neural_network.NeuralNetwork import NeuralNetwork
from CanvasManager import CanvasManager

def main():
    def load_image(image_path):
        img = PIL.Image.open(image_path)
        img = ImageOps.grayscale(img)
        img = img.resize(size=(28, 28))
        img = np.array(img)
        img = img.astype(float)
        img /= 255
        return img

    # DATA NEEDED FOR TRAINING PURPOSES
    # -----------------------------------------------------------------------------------------
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
    x_train = x_train.astype(float)
    x_train /= 255
    y_train = utils.to_categorical(y_train)

    model = NeuralNetwork(0.2)
    model.train(x_train[0:1000], y_train[0:1000], 20)
    # -----------------------------------------------------------------------------------------

    x = 0

    while x < 10:
        canvas_manager = CanvasManager()
        canvas_manager.create_new_canvas()

        img = load_image(f'x_train/digit.jpg')
        print()
        print(model.predict(img))

        os.system('pause')
        x += 1


if __name__ == '__main__':
    main()