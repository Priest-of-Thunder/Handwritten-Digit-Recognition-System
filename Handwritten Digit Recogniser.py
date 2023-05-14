import PIL
import numpy as np
from keras.models import load_model
from tkinter import *

# Load the model
model = load_model('digit_recognition.h5')

# Create the GUI
root = Tk()
root.title('Handwritten Digit Recognition')

# Define the canvas
canvas = Canvas(root, width=200, height=200, bg='white')
canvas.pack()

# Define the button
button_clear = Button(text='Clear', command=lambda: clear())
button_clear.pack(side='left')
button_recognize = Button(text='Recognize', command=lambda: recognize())
button_recognize.pack(side='right')

# Define the functions
def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill='black', width=10)

def clear():
    canvas.delete('all')

def recognize():
    canvas.postscript(file='digit.eps')
    img = PIL.Image.open('digit.eps').convert('L')
    img = img.resize((28, 28))
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32') / 255
    result = model.predict([img])[0]
    digit=np.argmax(result)
    confidence = round(result[digit]*100, 2)
    label_text.set(f"The recognized digit is: {digit}\n\nConfidence: {confidence}%")

# Bind the canvas
canvas.bind('<B1-Motion>', paint)

# Define the label
label_text = StringVar()
label = Label(root, textvariable=label_text)
label.pack()

# Run the GUI
root.mainloop()
