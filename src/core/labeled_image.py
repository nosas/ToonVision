import os

from PIL import Image

from src.core.bounding_box import BBFORMAT, BoundingBox
from src.dataset.constants import DEFAULT_OUTPUT_IMAGES_DIR


class LabeledImage:
    def __init__(self, filename: str, image: Image.Image, boxes: list[BoundingBox]):
        self.filename = filename
        self.image = image
        self.boxes = boxes

    def extract_bboxes(self, output_images_dir: str = DEFAULT_OUTPUT_IMAGES_DIR) -> None:
        """Extract bounding boxes from an image and save them to /output/images/bboxes

        Args:
            output_images_dir (str, optional): Output directory to save the extracted bounding boxes to. Defaults to OUTPUT_IMAGES_DIR.
            output_format (BBFORMAT, optional): Output format for the bounding boxes. Defaults to BBFORMAT.ABSOLUTE.
        """
        filename = self.filename.replace(".png", "")
        image = self.image
        boxes = self.boxes

        # Create the output directory if it does not exist
        os.makedirs(output_images_dir, exist_ok=True)

        for idx, box in enumerate(boxes):
            if box.format == BBFORMAT.YOLO:
                box = BoundingBox.yolo_to_absolute(
                    x=box.x,
                    y=box.y,
                    w=box.w,
                    h=box.h,
                    img_width=image.width,
                    img_height=image.height,
                    label=box.label,
                )

            bbox_image = image.crop(
                (int(box.x), int(box.y), int(box.x + box.w), int(box.y + box.h))
            )
            bbox_image.save(
                fp=os.path.join(output_images_dir, f"{filename}_{box.label}_{idx}.png"),
                format="PNG",
            )
