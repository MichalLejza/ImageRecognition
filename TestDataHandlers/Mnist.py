from DataHandlers.Mnist import MnistDataset


if __name__ == '__main__':
    dataset = MnistDataset(train=True, kind='Classic')
    print(dataset.__size__())
    print(len(dataset))
