from random import randint

import numpy as np
from PIL import Image, ImageDraw


def make_square():
    """
    Generates a square image with random size and position.

    Returns:
        np.array: A 16x16 grayscale image as a numpy array.
    """
    img = Image.new('L', (16, 16), 0)

    ulx = randint(1, 8)
    uly = randint(1, 8)
    lrx = randint(ulx, 16)
    lry = randint(uly, 16)

    draw = ImageDraw.Draw(img)
    draw.rectangle((ulx, uly, lrx, lry), fill=255)

    return np.array(img)


def make_circle():
    """
    Generates a circular image with random size and position.

    Returns:
        np.array: A 16x16 grayscale image as a numpy array.
    """
    img = Image.new('L', (16, 16), 0)

    ulx = randint(2, 13)
    uly = randint(2, 13)
    l = randint(5, 10)

    draw = ImageDraw.Draw(img)
    draw.ellipse((ulx-l/2, uly-l/2, ulx+l/2, uly+l/2), fill=255)

    return np.array(img)


def make_dataset(num_samples, split):
    """
    Generates a dataset of square and circular images with random size and position.

    Args:
        num_samples (int): The total number of samples to generate.
        split (float): The fraction of samples to use for training.

    Returns:
        tuple: A tuple of two tuples, each containing a dataset and its corresponding labels.
            The first tuple contains the training dataset and labels, while the second contains
            the validation dataset and labels.
    """
    dataset = []
    labels = []
    for _ in range(int(num_samples / 2)):
        dataset.append(make_square().reshape(16 * 16, ))
        labels.append(1)
        dataset.append(make_circle().reshape(16 * 16, ))
        labels.append(-1)

    dataset_np = np.stack(dataset, axis=0)
    labels_np = np.stack(labels)

    dataset_train = dataset_np[:int(num_samples * split)]
    labels_train = labels_np[:int(num_samples * split)]

    dataset_val = dataset_np[int(num_samples * split):]
    labels_val = labels_np[int(num_samples * split):]

    return (dataset_train, labels_train), (dataset_val, labels_val)