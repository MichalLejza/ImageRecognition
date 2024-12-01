from cnn import CNN


if __name__ == '__main__':
    model = CNN(batch_size=64, epochs=10, kind='Classic')
    model.train_model()

