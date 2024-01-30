from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BBFORMAT(Enum):
    YOLO = 0  # x, y (center of the image), w, h. Values are normalized to [0, 1]
    ABSOLUTE = 1  # x, y (top-left corner of the image), w, h. Values are absolute pixel values
    PIL = 2  # x1, y1, x2, y2 (left, top, right, bottom)


@dataclass
class BoundingBox:
    x: int | float
    y: int | float
    w: int | float
    h: int | float
    label: Optional[str] = None
    format: BBFORMAT = BBFORMAT.ABSOLUTE

    @staticmethod
    def yolo_to_absolute(
        x: float,
        y: float,
        w: float,
        h: float,
        img_width: int,
        img_height: int,
        label: Optional[str] = None,
    ) -> "BoundingBox":
        """Convert YOLO format (x, y, w, h) to absolute format

        The YOLO bounding box format consists of four values: (x, y, width, height).

        `x` and `y` represent the coordinates of the center of the bounding box, relative to the width and height of the image.
            These values range from 0 to 1, where (0, 0) represents the top-left corner of the image, and (1, 1) represents the bottom-right corner.
            `w` and `h` represent the width and height of the bounding box, also normalized relative to the image size.
        To convert these YOLO bounding box values to absolute values, you need to know the dimensions of the original image.

        Args:
            x (float): x-coordinate of the center of the bounding box, relative to the width of the image.
            y (float): y-coordinate of the center of the bounding box, relative to the height of the image.
            w (float): width of the bounding box, relative to the width of the image.
            h (float): height of the bounding box, relative to the height of the image.
            img_width (int): width of the image.
            img_height (int): height of the image.

        Returns:
            BoundingBox: BoundingBox containing the absolute coordinates for left, top, width, and height of the bbox
                (absolute_x, absolute_y, absolute_width, absolute_height)
        """
        absolute_x = int((x - w / 2) * img_width)
        absolute_y = int((y - h / 2) * img_height)
        absolute_width = int(w * img_width)
        absolute_height = int(h * img_height)

        return BoundingBox(
            x=absolute_x,
            y=absolute_y,
            w=absolute_width,
            h=absolute_height,
            label=label,
            format=BBFORMAT.ABSOLUTE,
        )

    @classmethod
    def from_yolo(cls, annotation: list[str]) -> "BoundingBox":
        """Create a BoundingBox from a YOLO annotation

        Args:
            annotation (list[str]): List of annotation values.
            Example input : ['6', '0.4023', '0.4336', '0.1077', '0.4699']
            Columns       : label, x, y, w, h

        Returns:
            BoundingBox: BoundingBox object in YOLO format
        """
        label = annotation[0]
        x, y, w, h = map(float, annotation[1:])

        return cls(
            x=x,
            y=y,
            w=w,
            h=h,
            label=label,
            format=BBFORMAT.YOLO,
        )
