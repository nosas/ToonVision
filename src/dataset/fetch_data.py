import os

from PIL import Image

from src.core.bounding_box import BoundingBox
from src.core.labeled_image import LabeledImage
from src.dataset.constants import DEFAULT_IMAGES_DIR, DEFAULT_LABELS_DIR


def fetch_image(input_images_dir: str, filename: str) -> Image.Image:
    """
    Fetches a particular image from the specified directory.

    Args:
        input_images_dir (str): The directory where the images are located.
        filename (str): The name of the image file to fetch.

    Returns:
        PIL.Image.Image: The fetched image.

    Raises:
        FileNotFoundError: If the specified image file does not exist.
    """
    return Image.open(fp=os.path.join(input_images_dir, filename))


def fetch_yolo_boxes(input_labels_dir: str, filename: str) -> list[BoundingBox]:
    """
    Fetch YOLO annotations (bounding boxes) from the specified directory for a particular image.

    Args:
        input_labels_dir (str): The directory path where the labels are stored.
        filename (str): The name of the file containing the labels.

    Returns:
        list[BoundingBox]: A list of BoundingBox objects representing the fetched bounding boxes.
    """
    boxes = []

    with open(os.path.join(input_labels_dir, filename)) as label_file:
        for line in label_file:
            boxes.append(BoundingBox.from_yolo(line.split()))

    return boxes


def fetch_labeled_image(
    filename: str,
    input_images_dir: str = DEFAULT_IMAGES_DIR,
    input_labels_dir: str = DEFAULT_LABELS_DIR,
) -> LabeledImage:
    """
    Fetches a labeled image.

    Args:
        filename (str): The name of the image file.
        input_images_dir (str, optional): The directory where the images are stored.
            Defaults to DEFAULT_IMAGES_DIR.
        input_labels_dir (str, optional): The directory where the labels are stored.
            Defaults to DEFAULT_LABELS_DIR.

    Returns:
        LabeledImage: The fetched labeled image.
    """
    image = fetch_image(input_images_dir=input_images_dir, filename=filename)
    boxes = fetch_yolo_boxes(
        input_labels_dir=input_labels_dir, filename=filename.replace(".png", ".txt")
    )

    return LabeledImage(filename=filename, image=image, boxes=boxes)


def fetch_all_images(input_images_dir: str) -> list[Image.Image]:
    """
    Fetches all the images with the ".png" extension from the specified directory.

    Args:
        input_images_dir (str): The directory path where the images are located.

    Returns:
        list[Image.Image]: A list of PIL Image objects representing the fetched images.
    """
    images = []

    for filename in os.listdir(input_images_dir):
        if filename.endswith(".png"):
            image = fetch_image(input_images_dir=input_images_dir, filename=filename)
            images.append(image)

    return images


def fetch_all_yolo_boxes(input_labels_dir: str) -> list[BoundingBox]:
    """
    Fetches all YOLO boxes from the given input_labels_dir.

    Args:
        input_labels_dir (str): The directory path where the YOLO labels are stored.

    Returns:
        list[BoundingBox]: A list of BoundingBox objects containing the fetched YOLO boxes.
    """
    boxes = []

    for filename in os.listdir(input_labels_dir):
        if filename.endswith(".txt"):
            boxes.extend(fetch_yolo_boxes(input_labels_dir=input_labels_dir, filename=filename))

    return boxes


def fetch_all_labeled_images(
    input_images_dir: str = DEFAULT_IMAGES_DIR, input_labels_dir: str = DEFAULT_LABELS_DIR
) -> list[LabeledImage]:
    """
    Fetches all labeled images from the specified input directories.

    Args:
        input_images_dir (str): The directory path where the input images are stored.
            Defaults to DEFAULT_IMAGES_DIR.
        input_labels_dir (str): The directory path where the input labels are stored.
            Defaults to DEFAULT_LABELS_DIR.

    Returns:
        list[LabeledImage]: A list of LabeledImage objects containing the fetched images
        and their corresponding bounding boxes.
    """
    labeled_images = []

    for filename in os.listdir(input_images_dir):
        if filename.endswith(".png"):
            labeled_images.append(
                fetch_labeled_image(
                    filename=filename,
                    input_images_dir=input_images_dir,
                    input_labels_dir=input_labels_dir,
                )
            )

    return labeled_images
