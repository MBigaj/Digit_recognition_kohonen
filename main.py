from PIL import Image, ImageOps
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

(data, templates), (test_data, test_templates) = mnist.load_data()
data = data.astype(float)
data /= 255
templates = np_utils.to_categorical(templates)