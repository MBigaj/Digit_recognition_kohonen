from PIL import Image, ImageOps
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from Neural_Network import Network
import matplotlib.pyplot as plt
import tensorflow as tf

def load_image(image_name):
    img = Image.open(image_name)
    img = ImageOps.grayscale(img)
    img = img.resize(size=(28, 28))
    img = np.array(img)
    img = img.astype(float)
    img /= 255
    return img

(data, templates), (test_data, test_templates) = mnist.load_data()
data = data.reshape(data.shape[0], 1, 28*28)
data = data.astype(float)
data /= 255
templates = np_utils.to_categorical(templates)

network = Network(0.1)
network.train(data[0:1000], templates[0:1000], 35)

img = load_image('digit.jpg')
print()
print(network.predict(img))