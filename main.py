from PIL import ImageOps, ImageDraw
import PIL.Image
import PIL
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from Neural_Network import Network
from tkinter import *
import tkinter as tk
import matplotlib.pyplot as plt

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
(data, templates), (test_data, test_templates) = mnist.load_data()
data = data.reshape(data.shape[0], 1, 28*28)
data = data.astype(float)
data /= 255
templates = np_utils.to_categorical(templates)

network = Network(0.2)
network.train(data[0:1000], templates[0:1000], 30)
# -----------------------------------------------------------------------------------------


def save_image():
    drawn_image.save(f'data/digit.jpg')
    root.destroy()

def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    new_canvas.create_oval(x1, y1, x2, y2, fill='black', width=16)
    draw.rounded_rectangle([x1, y1, x2, y2], fill='black', width=10)

root = tk.Tk()
new_canvas = tk.Canvas(root, bg='white', width=400, height=400)
new_canvas.pack()

drawn_image = PIL.Image.new('RGB', (400, 400), 'white')
draw = ImageDraw.Draw(drawn_image)

new_canvas.pack(fill='both', expand='yes')
new_canvas.bind('<B1-Motion>', paint)

button = Button(text='Predict', command=save_image)
button.pack()

root.mainloop()

img = load_image('data/digit.jpg')
print()
print(network.predict(img))