from DataHandlers import MnistDataset


if __name__ == '__main__':
    dataset = MnistDataset(train=True, kind='ByClass')
    print(dataset)
