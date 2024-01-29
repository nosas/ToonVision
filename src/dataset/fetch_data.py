import os

from PIL import Image

from src.core.bounding_box import BoundingBox
from src.core.labeled_image import LabeledImage
from src.dataset.constants import DEFAULT_IMAGES_DIR, DEFAULT_LABELS_DIR


def fetch_image(input_images_dir: str, filename: str) -> Image.Image:
    """Fetch a particular image from /data/images"""
    return Image.open(fp=os.path.join(input_images_dir, filename))


def fetch_yolo_boxes(input_labels_dir: str, filename: str) -> list[BoundingBox]:
    """Fetch bounding boxes from /data/labels for a particular image"""
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
    """Fetch a particular image and its bounding boxes from /data/images and /data/labels"""
    image = fetch_image(input_images_dir=input_images_dir, filename=filename)
    boxes = fetch_yolo_boxes(
        input_labels_dir=input_labels_dir, filename=filename.replace(".png", ".txt")
    )

    return LabeledImage(filename=filename, image=image, boxes=boxes)


def fetch_all_images(input_images_dir: str) -> list[Image.Image]:
    """Fetch images from /data/images"""
    images = []

    for filename in os.listdir(input_images_dir):
        if filename.endswith(".png"):
            image = fetch_image(input_images_dir=input_images_dir, filename=filename)
            images.append(image)

    return images


def fetch_all_yolo_boxes(input_labels_dir: str) -> list[BoundingBox]:
    """Fetch all bounding boxes from /data/labels"""
    boxes = []

    for filename in os.listdir(input_labels_dir):
        if filename.endswith(".txt"):
            boxes.extend(fetch_yolo_boxes(input_labels_dir=input_labels_dir, filename=filename))

    return boxes


def fetch_all_labeled_images(
    input_images_dir: str = DEFAULT_IMAGES_DIR, input_labels_dir: str = DEFAULT_LABELS_DIR
) -> list[LabeledImage]:
    """Fetch images and bounding boxes from /data/images and /data/labels"""
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
