from cnn import CNN


if __name__ == '__main__':
    model = CNN(batch_size=64, epochs=10, kind='MNIST')
    print(model)
    model.test_set.plotEightImages()
    model.train_model()

