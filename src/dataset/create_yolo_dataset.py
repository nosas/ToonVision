"""Library to create a YOLO training dataset from a directory of images and a directory of annotations.

Example:
    $ python -m src.dataset.create_yolo_dataset --input_images_dir data/images --input_labels_dir data/labels --output_dir output/datasets/entity_only

Output directory structure:
    <output_dir>/
        dataset.yaml
        images/
            train/
                <id1>.<ext>
                <id2>.<ext>
                ...
            val/
                <id3>.<ext>
                <id4>.<ext>
                ...
        labels/
            train/
                <id1>.txt
                <id2>.txt
                ...
            val/
                <id3>.txt
                <id4>.txt
                ...
"""

import argparse
import logging
import os
import random
from typing import Optional

import yaml

from src.dataset.constants import (
    DEFAULT_IMAGES_DIR,
    DEFAULT_LABELS_DIR,
    DEFAULT_OUTPUT_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Create a YOLO training dataset from a directory of images and a directory of annotations."
    )
    parser.add_argument(
        "--input_images_dir",
        type=str,
        default=DEFAULT_IMAGES_DIR,
        help="Path to directory containing input images.",
    )
    parser.add_argument(
        "--input_labels_dir",
        type=str,
        default=DEFAULT_LABELS_DIR,
        help="Path to directory containing input labels.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Path to directory where output dataset will be created.",
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.8,
        help="Fraction of images to use for training (remainder will be used for validation).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generator.",
    )

    return parser.parse_args()


def replace_class_label(
    file_path: str, replacement_class: str, target_class: Optional[str] = None
) -> None:
    """
    Replace the class label in each line of a YOLO annotation file.

    Args:
        file_path (str): Path to label file.
        replacement_class (str): Class label to replace with.
        target_class (Optional[str], optional): Class label to replace. Defaults to None.

    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    with open(file_path, "w") as file:
        for line in lines:
            parts = line.split()
            if parts:
                if target_class is None or parts[0] == target_class:
                    parts[0] = replacement_class
                file.write(" ".join(parts) + "\n")


def create_dataset(
    input_images_dir: str,
    input_labels_dir: str,
    output_dir: str,
    train_val_split: float,
    seed: Optional[int] = None,
) -> None:
    """Create a YOLO training dataset from a directory of images and a directory of annotations.

    Args:
        input_images_dir (str): Path to directory containing input images.
        input_labels_dir (str): Path to directory containing input labels.
        output_dir (str): Path to directory where output dataset will be created.
        train_val_split (float): Fraction of images to use for training (remainder will be used for validation).
        seed (Optional[int]): Seed for random number generator.
    """
    # Set seed for random number generator
    random.seed(seed)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)

    # Get list of all images
    images = []
    for filename in os.listdir(input_images_dir):
        if filename.endswith(".png"):
            images.append(filename)

    # Shuffle images
    random.shuffle(images)

    # Split into train and val sets
    num_train = int(len(images) * train_val_split)
    train_images = images[:num_train]
    val_images = images[num_train:]

    # Copy images to output directory
    for image in train_images:
        os.system(
            f"cp {os.path.join(input_images_dir, image)} {os.path.join(output_dir, 'images', 'train')}"
        )
    for image in val_images:
        os.system(
            f"cp {os.path.join(input_images_dir, image)} {os.path.join(output_dir, 'images', 'val')}"
        )

    # Copy labels to output directory
    for image in train_images:
        os.system(
            f"cp {os.path.join(input_labels_dir, image.replace('.png', '.txt'))} {os.path.join(output_dir, 'labels', 'train')}"
        )
    for image in val_images:
        os.system(
            f"cp {os.path.join(input_labels_dir, image.replace('.png', '.txt'))} {os.path.join(output_dir, 'labels', 'val')}"
        )

    # Replace all labels with "0" (entity)
    for filename in os.listdir(os.path.join(output_dir, "labels", "train")):
        replace_class_label(
            file_path=os.path.join(output_dir, "labels", "train", filename),
            replacement_class="0",
        )

    for filename in os.listdir(os.path.join(output_dir, "labels", "val")):
        replace_class_label(
            file_path=os.path.join(output_dir, "labels", "val", filename),
            replacement_class="0",
        )

    # Create dataset.yaml
    dataset = {
        "path": output_dir,
        "train": "images/train",
        "val": "images/val",
        "names": {0: "entity"},
    }

    with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
        yaml.dump(dataset, f, default_flow_style=False)


if __name__ == "__main__":
    args = parse_args()

    create_dataset(
        input_images_dir=args.input_images_dir,
        input_labels_dir=args.input_labels_dir,
        output_dir=args.output_dir,
        train_val_split=args.train_val_split,
        seed=args.seed,
    )
