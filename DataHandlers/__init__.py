import os
import platform

# acknowledging different OS
OS_NAME = platform.system()

# Creating Datapath to Datasets according to OS
if OS_NAME == "Windows":  # Windows
    DATA_PATH = os.path.expanduser("~\\Desktop\\Data\\ImageClassification")
elif OS_NAME == "Darwin":  # macos
    DATA_PATH = os.path.expanduser("~/Desktop/Data/ImageClassification")
else:
    raise ValueError("Your Operating Systems is not supported!\nTry Windows/MacOS")

# List of currently all available Datasets on this Machine
AVAILABLE_DATASETS = [_sets for _sets in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, _sets))]

SLASH = '/' if OS_NAME == 'Darwin' else '\\'


def get_dataset_path(dataset_name: str) -> str:
    """
    Method checks if there exists a Dataset and returns its path.
    :param dataset_name: Name of Dataset as string.
    :return: Path to dataset as string.
    """
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset: {dataset_name} is currently not available. {AVAILABLE_DATASETS}")
    return os.path.join(DATA_PATH, dataset_name)


# Exported elements of module
__all__ = [
    "OS_NAME",
    "DATA_PATH",
    "AVAILABLE_DATASETS",
    "get_dataset_path",
    "SLASH"
]
