# coding=utf8

import os
import re
from typing import Dict

from lark import Lark, Token
from lark import Tree as LarkTree
from lark.visitors import Visitor

from utils import Element, ElementStyle, Region

from . import utils as ir_utils

MULTIINFO_REGION = 'MultiInfo'
SINGLEINFO_REGION = 'SingleInfo'
FOOTER_REGION = 'Footer'
HEADER_REGION = 'Header'
REGIONS = [MULTIINFO_REGION, SINGLEINFO_REGION, FOOTER_REGION, HEADER_REGION]

HIGHLIGHT_TEXT_LIKE_ELEMENTS = {"title"}
TEXT_LIKE_ELEMENTS = {"text", "button", "link", "title", "description"}
IMAGE_LIKE_ELEMENTS = {"image", "background_image", "icon", "logo"}
NAV_LINK_ELEMENTS = {"link", "button"}


def get_parser(grammar_file):
    curr_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(curr_path, grammar_file), "r") as f:
        grammar = f.read()
    return Lark(grammar, maybe_placeholders=False)


class IRToRegion(Visitor):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._region = None
        self.elements = list()

    def region(self, tree: LarkTree):
        rtype, bbox = None, None
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'ELEMENT_TYPE':
                rtype = child.value
            elif isinstance(child, LarkTree) and child.data == 'attr':
                if child.children[0].value == 'bbox':
                    bbox_str = child.children[1].value
                    bbox = [float(v) for v in bbox_str.strip("'").split(",")]
        self._region = Region(rtype, bbox[2], bbox[3], self.elements)

    def element(self, tree: LarkTree):
        # element type, position, text, font_size
        etype = ir_utils.get_element_type(tree)
        bbox, value, estyle = None, None, ElementStyle()
        for child in tree.children:
            if not ir_utils.is_attr(child):
                continue
            attr_type = ir_utils.get_attr_type(child)
            attr_value = child.children[1].value.strip("'")
            if attr_type == 'bbox':
                bbox = [float(v) for v in attr_value.split(",")]
            elif attr_type == 'value':
                value = attr_value
            elif attr_type.startswith('style_'):
                style_key = re.match(r'^style_(.*)', attr_type).group(1)
                setattr(estyle, style_key, attr_value)

        if etype in TEXT_LIKE_ELEMENTS:
            self.elements.append(Element(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3],
                                         etype=etype, text=value, image=None, style=estyle))
        else:
            self.elements.append(Element(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3],
                                         etype=etype, text=None, image=value, style=estyle))


def ir_to_region(pt: LarkTree) -> Region:
    func = IRToRegion()
    func.visit(pt)
    return func._region


def ir_to_json(pt: LarkTree) -> Dict:

    def _visit(tree: LarkTree):
        if ir_utils.is_element(tree):
            value, bbox, styles = None, None, dict()
            for child in tree.children:
                if not ir_utils.is_attr(child):
                    continue
                attr_type = ir_utils.get_attr_type(child)
                attr_value = child.children[1].value.strip("'")
                if attr_type == 'value':
                    value = attr_value
                elif attr_type == 'bbox':
                    bbox = [float(v) for v in attr_value.split(",")]
                elif attr_type.startswith('style_'):
                    style_key = re.match(r'^style_(.*)', attr_type).group(1).replace("_", "-")
                    if style_key.endswith("-size") or style_key.endswith("-width") or style_key.endswith("-height"):
                        attr_value = f'{attr_value}px'
                    styles[style_key] = attr_value
            result = {
                "type": ir_utils.get_element_type(tree),
                "value": value,
                "b_box": bbox,
                "styles": styles
            }
            if result['type'] == "input" and isinstance(value, str):
                result['value'] = value.lower()
            return result

        result = {
            "type": "", "children": list()
        }
        if ir_utils.is_region(tree):
            result['type'] = ir_utils.get_region_type(tree)
            bbox = None
            for child in tree.children:
                if ir_utils.is_target_attr(child, 'bbox'):
                    attr_value = child.children[1].value.strip("'")
                    bbox = [float(v) for v in attr_value.split(",")]
                    break
            result['b_box'] = bbox
        elif ir_utils.is_group(tree):
            result['type'] = ir_utils.get_group_type(tree)
        elif ir_utils.is_item(tree):
            result['type'] = "item"
        else:
            return None
        for child in tree.children:
            child_result = _visit(child)
            if child_result is not None:
                result['children'].append(child_result)
        return result

    out = _visit(pt.children[0])
    return out
