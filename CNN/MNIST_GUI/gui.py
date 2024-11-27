import subprocess
import cv2
from tkinter import *

import numpy as np
import torch

from cnn import CNN


class GUI(object):
    """
    Klasa GUI tworzy prostą aplikację GUI dzięki bibliotece tkinter i dzięki modelowi sztucznej inteligencji
    pokazuje Liczbę albo dużą Literę którą użytkownik napisał w polu do rysowania
    """

    def __init__(self, model: CNN):
        """
        Konstroktor klasy. Tworzy wszystkie elementy i ustawia je w odpowiednim miejscu
        :param model: Wytrenowany perceptron
        """
        self.model = model
        self.root = Tk()
        # Label informacyjny
        self.title = Label(self.root, text="MNIST Recogniser", font=("Helvetica", 20))
        self.title.grid(row=0, column=4, columnspan=2)
        # Pole w którym będziemy rysować Liczbę/Literę
        self.canvas = Canvas(self.root, bg='white', width=350, height=350)
        self.canvas.grid(row=1, column=0, rowspan=5, columnspan=5)
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)
        # przycisk do przetworzenia canvasu na tablicę i przewidzenia znaku który narysowaliśmy
        self.predictButton = Button(self.root, text='predict', command=self.predictImage)
        self.predictButton.grid(row=3, column=6)
        # przycisk który czyści canvas
        self.clearButton = Button(self.root, text='clear', command=self.clearCanvas)
        self.clearButton.grid(row=3, column=7)
        # pole tekstowe do wyświetlania przewidzianych znaków
        self.textField = Text(self.root, font=("Helvetica", 28), height=2, width=15)
        self.textField.grid(row=2, column=6, columnspan=3)
        # przycisk do cofania ostatniego przewidzianego znaku
        self.backSpaceButton = Button(self.root, text='backspace', command=self.backSpace)
        self.backSpaceButton.grid(row=3, column=8)

        self.oldX = None
        self.oldY = None
        self.root.mainloop()

    def clearCanvas(self) -> None:
        """
        Metoda czyści Canvas
        :return: None
        """
        self.canvas.delete('all')

    def backSpace(self) -> None:
        """
        Metoda cofa ostatni przewidziany znak
        :return: None
        """
        currentText = self.textField.get('1.0', 'end')
        if currentText.strip():
            self.textField.delete("end - 2 chars", END)

    def predictImage(self) -> None:
        """
        Metoda transformuje canvas do pliku .png a następnie na tablicę 28x28
        :return: None
        """
        self.canvas.postscript(file="Models/canvas.ps", colormode='color')
        subprocess.run(['gs', '-o', 'Models/canvas.png', '-sDEVICE=pngalpha', '-r144', 'Models/canvas.ps'])
        imageFromCanvas = cv2.imread('Models/canvas.png')
        imageHeight, imageWidth, _ = imageFromCanvas.shape
        croppedImage = imageFromCanvas[430:imageHeight - 430, 250:imageWidth - 250]
        grayScaleImage = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)
        resizedImage = cv2.resize(grayScaleImage, (28, 28), interpolation=cv2.INTER_AREA)
        imageArray = np.array(resizedImage)
        imageArray = np.abs((imageArray / 255.0) - 1)
        image = torch.tensor(imageArray)
        image.reshape(shape=(1, 28, 28))
        predictedLabel = self.model.predict(image)
        self.textField.insert(END, predictedLabel.numpy()[0])
        self.clearCanvas()

    def paint(self, event) -> None:
        """
        Metoda do malowania na Canvasie
        :param event:
        :return: None
        """
        if self.oldX and self.oldY:
            self.canvas.create_line(self.oldX, self.oldY, event.x, event.y, width=36, fill='black', capstyle=ROUND,
                                    smooth=TRUE, splinesteps=36)
        self.oldX = event.x
        self.oldY = event.y

    def reset(self, event) -> None:
        """
        Metoda która sprawia że nie rysujemy gdy nie wciskamy przycisku
        :param event:
        :return: None
        """
        self.oldX, self.oldY = None, None
