# coding=utf8

import copy
import re
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import List, Tuple, Union

from lark import Token
from lark import Tree as LarkTree
from lark.visitors import Transformer

from .parser import get_parser
from .utils import (create_attr, create_element, get_attr, is_element,
                    is_target_attr)

PlacementInput = namedtuple('PlacementInput', ['pt', 'input', 'emap'])


def is_num(value: str) -> bool:
    return re.match(r'\d+', value)


class GroupExpansion(Transformer):

    DEFAULT_ITEM_REPEAT = 4
    DEFAULT_NAV_TEXT_VALUE = 'Navigation'
    DEFAULT_ELEMENT_REPEAT = 4

    def _get_flat_nav_item_template(self, name: str = None, element_type: str = "link") -> LarkTree:
        _name = name or self.DEFAULT_NAV_TEXT_VALUE
        text_element = create_element(element_type, [('value', _name)])
        return LarkTree("item", [text_element])

    def group(self, children):
        # Handle repeat
        num_repeat, items = 0, list()
        for child in children:
            if isinstance(child, LarkTree):
                if child.data == 'item':
                    items.append(child)
                elif child.data == 'group_attr':
                    attr_name, attr_value = child.children[0].value, child.children[1].value.strip("'")
                    if attr_name == 'repeat':
                        num_repeat = int(attr_value) if is_num(attr_value) else self.DEFAULT_ITEM_REPEAT

        if num_repeat > 1 and len(items) == 1:
            children.extend([copy.deepcopy(items[0])
                            for i in range(num_repeat - 1)])
        elif num_repeat > 0 and len(items) == 0:
            for i in range(num_repeat):
                children.append(self._get_flat_nav_item_template())

        return LarkTree("group", children)


class AddElementId(Transformer):

    def __init__(self, visit_tokens: bool = True) -> None:
        super().__init__(visit_tokens)
        self._id = 0

    def element(self, children):
        attr = create_attr('eid', f'e_{self._id}')
        children.append(attr)
        self._id += 1
        return LarkTree('element', children)


@dataclass
class Element:
    etype: str
    eid: str
    pos: str
    size: str

    ELEMENT_MAP_TYPE = {
        'text': 'text',
        'image': 'image',
        'icon': 'icon',
        'list item': 'list item',
        'text button': 'text button',
        'toolbar': 'toolbar',
        'web view': 'web view',
        'input': 'input',
        'card': 'card',
        'advertisement': 'advertisement',
        'background image': 'background image',
        'drawer': 'drawer',
        'radio button': 'radio button',
        'checkbox': 'checkbox',
        'multi-tab': 'multi-tab',
        'pager indicator': 'pager indicator',
        'modal': 'modal',
        'on/off switch': 'on/off switch',
        'slider': 'slider',
        'map view': 'map view',
        'button bar': 'button bar',
        'video': 'video',
        'bottom navigation': 'bottom navigation',
        'number stepper': 'number stepper',
        'date picker': 'date picker',
    }

    POS_PRIORITY = {
        "top": 0,
        "bottom": 1,
        "left": 2,
        "right": 3,
        "center": 4,
        "undefined": 5
    }
    POS_MAP = {
        "top": "top",
        "bottom": "bottom",
        "left": "left",
        "right": "right",
        "middle": "center",
        "top-left": "top",
        "top-right": "top"
    }

    SIZE_PRIORITY = {"small": 0, "large": 1, "undefined": 2}

    def __post_init__(self):
        self.petype = self.ELEMENT_MAP_TYPE.get(self.etype)
        self.ppos = self.POS_MAP.get(self.pos, self.pos)
        self.pos_priority = self.POS_PRIORITY.get(self.pos)
        self.size_priority = self.SIZE_PRIORITY.get(self.size)

    def __str__(self) -> str:
        return f"{self.petype} {self.ppos} {self.size}"

    @staticmethod
    def sort_elements(elements):
        elements.sort(key = lambda e: (e.petype, e.pos_priority, e.size_priority))


class ToPlacementInput:

    PLACEHOLDER = 'undefined'

    def linearize_element(self, element: LarkTree) -> Element:
        etype = element.children[0].value
        eid, pos, size = None, self.PLACEHOLDER, self.PLACEHOLDER
        for child in element.children:
            if isinstance(child, LarkTree) and child.data == 'attr':
                attr_type, attr_value = child.children[0].value, child.children[1].value.strip("'")
                if attr_type == 'position':
                    pos = attr_value
                elif attr_type == 'size':
                    size = attr_value
                elif attr_type == 'eid':
                    eid = attr_value
        return Element(etype, eid, pos, size)

    def linearize_group(self, group: LarkTree) -> List[Tuple[List[Element], bool]]:
        result = list()
        gp_size, gp_pos = 'undefined', 'undefined'

        for child in group.children:
            if isinstance(child, LarkTree) and child.data == 'item':
                result.append(self.linearize_item(child))
            if isinstance(child, Token):
                gp_type = child
            elif is_target_attr(child, 'size'):
                gp_size = get_attr(child).replace("'", '').replace('"', '')
            elif is_target_attr(child, 'position'):
                gp_pos = get_attr(child).replace("'", '').replace('"', '')

        results = [(result, gp_type, gp_size, gp_pos)]
        return results

    def linearize_item(self, item: LarkTree) -> List[Element]:
        elements = list()
        for child in item.children:
            if isinstance(child, LarkTree) and child.data == 'element':
                elements.append(self.linearize_element(child))
        Element.sort_elements(elements)
        # return "[{}]".format(" | ".join(elements))
        return elements

    def __call__(self, pt: LarkTree) -> Tuple:
        self.element_index = 0

        region_tree = pt.children[0]
        region_type = region_tree.children[0].value.lower()

        region_group = list()
        for child in region_tree.children:
            if isinstance(child, LarkTree):
                if child.data == 'element':
                    region_group.append(self.linearize_element(child))
                elif child.data == 'group':
                    # tuple
                    region_group.extend(self.linearize_group(child))

        elements, groups = list(), list()
        for i in region_group:
            if isinstance(i, Element):
                elements.append(i)
            else:
                groups.append(i)
        Element.sort_elements(elements)

        num_element, element_map = 0, dict()
        output = list()
        for e in elements:
            output.append(str(e))
            element_map[e.eid] = num_element
            num_element += 1

        for group, gp_type, gp_size, gp_pos in groups:
            parent = Element(etype=gp_type, eid=None, pos=gp_pos, size=gp_size)
            for item_idx, item in enumerate(group):
                item_output = list()
                item_output.append(str(parent))
                for e in item:
                    item_output.append(str(e))
                    element_map[e.eid] = num_element
                    num_element += 1
                output.append("[ {} ]".format(" | ".join(item_output)))

        input_str = "{} : {}".format(region_type, " | ".join(output))
        return input_str, element_map


class Executor:

    DEFAULT_ELEMENT_REPEAT = 4

    def __init__(self, grammar_file) -> None:
        self._parser = get_parser(grammar_file)
        self._placement_fn = ToPlacementInput()

    def _expand_group(self, pt: LarkTree) -> LarkTree:
        return GroupExpansion(visit_tokens=False).transform(pt)

    def _expand_element(self, pt: LarkTree) -> LarkTree:
        queue = deque()
        queue.append(pt)
        while len(queue) > 0:
            node = queue.popleft()
            new_elements = list()
            for child in node.children:
                if is_element(child):
                    repeat_attr_node, num_repeat = None, 1
                    for attr_node in child.children:
                        if is_target_attr(attr_node, 'repeat'):
                            repeat_attr_node = repeat_attr_node
                            attr_value = get_attr(attr_node).strip("'")
                            num_repeat = int(attr_value) if is_num(attr_value) else self.DEFAULT_ELEMENT_REPEAT
                            break
                    if repeat_attr_node is not None:
                        child.children.remove(repeat_attr_node)
                    if num_repeat > 1:
                        for i in range(num_repeat - 1):
                            new_elements.append(copy.deepcopy(child))
                elif isinstance(child, LarkTree):
                    queue.append(child)
            if len(new_elements) > 0:
                node.children.extend(new_elements)

    def _add_element_id(self, pt: LarkTree) -> LarkTree:
        return AddElementId(visit_tokens=False).transform(pt)

    def to_placement_input(self, lf: Union[str, LarkTree]) -> PlacementInput:
        if isinstance(lf, str):
            pt = self._parser.parse(lf)
        else:
            pt = lf
        self._expand_element(pt)
        pt_with_group_expansion = self._expand_group(pt)
        pt_with_eid = self._add_element_id(pt_with_group_expansion)
        input_str, emap = self._placement_fn(pt_with_eid)

        return PlacementInput(pt_with_eid, input_str, emap)

    def __call__(self, lf: Union[str, LarkTree]) -> List[PlacementInput]:
        if isinstance(lf, str):
            pt = self._parser.parse(lf)
        else:
            pt = lf

        pts = [pt]

        return [self.to_placement_input(_pt) for _pt in pts]
