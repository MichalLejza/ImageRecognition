from torchvision.transforms import transforms

def get_mnist_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Powiększenie obrazu do 224x224
        # transforms.Grayscale(num_output_channels=3),  # Konwersja 1-kanałowego obrazu na 3-kanałowy
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def get_cifar_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

def get_caltech_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

def get_imagenet_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])