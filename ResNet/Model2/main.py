from DataHandlers.Cifar10 import Cifar10Dataset


if __name__ == '__main__':
    dataSet = Cifar10Dataset(test=True)
    print(dataSet.__size__())
    dataset = Cifar10Dataset(train=True)
    print(dataset.__size__())
