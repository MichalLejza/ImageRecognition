from Models.DataHandlers.Mnist.Mnist import MnistDataset


if __name__ == '__main__':
    train = MnistDataset("ByClass/train-images", "ByClass/train-labels")

    train.printSize()
    train.plotImage(0)
    train.plotEightImages()
