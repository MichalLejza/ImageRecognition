import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split


def prepare_caltech256_splits(root_dir, output_dir, split_ratio=0.8, random_state=42):
    """
    Tworzy podział na zbiory treningowy i walidacyjny dla Caltech256,
    kopiując pliki do folderów 'train' i 'val'.

    :param root_dir: Ścieżka do oryginalnych danych (zawierających foldery klas).
    :param output_dir: Ścieżka do folderu, gdzie zostaną utworzone 'train' i 'val'.
    :param split_ratio: Procent danych treningowych.
    :param random_state: Seed dla powtarzalnego podziału.
    """

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")

    # Tworzenie katalogów docelowych
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Przejście przez wszystkie klasy
    class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for class_name in class_dirs:
        class_path = os.path.join(root_dir, class_name)
        images = glob(os.path.join(class_path, "*.jpg"))  # Zakładam format .jpg

        # Podział na trening i walidację
        train_images, val_images = train_test_split(images, train_size=split_ratio, random_state=random_state)

        # Tworzenie folderów dla klasy
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Kopiowanie plików
        for img_path in train_images:
            shutil.copy(img_path, os.path.join(train_class_dir, os.path.basename(img_path)))

        for img_path in val_images:
            shutil.copy(img_path, os.path.join(val_class_dir, os.path.basename(img_path)))

        print(f"Przetworzono klasę {class_name}: {len(train_images)} do train, {len(val_images)} do val.")


# Przykładowe użycie:
prepare_caltech256_splits("C:\\Users\\Michał\\Desktop\\Data\\ImageClassification\\CALTECH256",
                          "C:\\Users\\Michał\\Desktop\\Data\\ImageClassification")
