# dataset.py

from torchvision import datasets
from torchvision import transforms

from sklearn.model_selection import train_test_split

from config import RANDOM_SEED
from config import IMAGE_SIZE


def get_train_transform():

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(10),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_test_transform():

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_dataset(dataset_path):
    """
    Load AugmentedAlzheimerDataset
    """

    dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=get_train_transform()
    )

    return dataset


def create_train_test_split(
    dataset,
    train_ratio=0.8
):
    """
    Returns train indices and test indices.

    We do NOT use random_split().
    This avoids nested Subset problems later.
    """

    indices = list(
        range(len(dataset))
    )

    labels = dataset.targets

    train_indices, test_indices = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=labels,
        random_state=RANDOM_SEED
    )

    return train_indices, test_indices


def get_class_names(dataset):

    return dataset.classes


def get_num_classes(dataset):

    return len(dataset.classes)


def get_labels(dataset):

    return dataset.targets