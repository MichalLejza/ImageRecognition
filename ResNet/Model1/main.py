from ResNet import *


if __name__ == '__main__':
    model = ResNet()
    while True:
        i = int(input("Podaj index: "))
        model.train_data.displayImage(i)
