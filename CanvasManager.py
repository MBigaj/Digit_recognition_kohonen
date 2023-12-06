import os
import tkinter as tk
import PIL.Image
import PIL
from PIL import ImageDraw

class CanvasManager:
    root = None
    new_canvas = None
    draw = None
    drawn_image = None

    def __init__(self) -> None:
        if 'data' not in os.listdir():
            os.mkdir('data')

        self.root = tk.Tk()

    def create_new_canvas(self):
        self.new_canvas = tk.Canvas(self.root, bg='white', width=400, height=400)
        self.new_canvas.pack()

        self.drawn_image = PIL.Image.new('RGB', (400, 400), 'white')
        self.draw = ImageDraw.Draw(self.drawn_image)

        self.new_canvas.pack(fill='both', expand='yes')
        self.new_canvas.bind('<B1-Motion>', self.paint)

        button = tk.Button(text='Predict', command=self.save_image)
        button.pack()

        self.root.mainloop()

    def save_image(self):
        self.drawn_image.save(f'data/digit.jpg')
        self.root.destroy()


    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.new_canvas.create_oval(x1, y1, x2, y2, fill='black', width=16)
        self.draw.rounded_rectangle([x1, y1, x2, y2], fill='black', width=10)