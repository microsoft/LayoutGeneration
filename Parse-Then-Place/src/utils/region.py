# coding=utf8

from typing import ClassVar
from dataclasses import dataclass

import yaml


@dataclass
class Region:
    rtype: str
    width: int
    height: int
    elements: list


@dataclass
class ElementStyle:

    # Font Style
    font_size: int = None
    font_family: str = None
    text_align: str = None
    text_vertical_align: str = None
    text_decoration: str = None
    text_color: str = None

    TEXT_LEFT_ALIGN: ClassVar[str] = 'left'
    TEXT_RIGHT_ALIGN: ClassVar[str] = 'right'
    TEXT_CENTRE_ALIGN: ClassVar[str] = 'centre'

    TEXT_VERTICAL_TOP_ALIGN: ClassVar[str] = 'top'
    TEXT_VERTICAL_MIDDLE_ALIGN: ClassVar[str] = 'middle'
    TEXT_VERTICAL_BOTTOM_ALIGN: ClassVar[str] = 'bottom'

    TEXT_DECORATION_UNDERLINE: ClassVar[str] = 'underline'

    # Border Style
    border: str = None
    border_weight: int = 0
    border_color: str = None


@dataclass
class Element:
    x: int
    y: int
    width: int
    height: int
    etype: str
    text: str = None
    image: str = None
    style: ElementStyle = None

    def __post_init__(self):
        if self.style is None:
            self.style = ElementStyle()

        # Update default style according to element type
        if self.style.text_align is None:
            if self.etype in {"button"}:
                self.style.text_align = ElementStyle.TEXT_CENTRE_ALIGN
            else:
                self.style.text_align = ElementStyle.TEXT_LEFT_ALIGN
        if self.style.text_vertical_align is None:
            if self.etype in {"button"}:
                self.style.text_vertical_align = ElementStyle.TEXT_VERTICAL_MIDDLE_ALIGN
            else:
                self.style.text_vertical_align = ElementStyle.TEXT_VERTICAL_TOP_ALIGN

    def get_ltwh_bbox(self):
        return [self.x, self.y, self.width, self.height]

    def get_ltrb_bbox(self):
        right, bottom = self.x + self.width, self.y + self.height
        return [self.x, self.y, right, bottom]

    def has_image(self):
        return self.image is not None and isinstance(self.image, str) and self.image.startswith("https:")

    def has_text(self):
        return self.text is not None and isinstance(self.text, str) and len(self.text) > 0


def read_region_bbox(path: str) -> Region:
    with open(path, 'r') as f:
        region = yaml.load(f)

    elements = list()
    for e in region['elements']:
        epos = e['position']
        elements.append(Element(epos[0], epos[1], epos[2], epos[3], e['type'],
                                e['text'], e['image']))
    region_pos = region['position']
    region_width, region_height = region_pos[2], region_pos[3]
    return Region(None, region_width, region_height, elements)
