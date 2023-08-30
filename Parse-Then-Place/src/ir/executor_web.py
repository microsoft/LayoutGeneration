# coding=utf8

import copy
import re
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import List, Set, Tuple, Union

from lark import Tree as LarkTree
from lark.visitors import Transformer

from .parser import TEXT_LIKE_ELEMENTS, get_parser
from .utils import (create_attr, create_element, get_attr, get_element_type,
                    is_element, is_item, is_target_attr, is_target_element)

PlacementInput = namedtuple('PlacementInput', ['pt', 'input', 'emap'])


def is_num(value: str) -> bool:
    return re.match(r'\d+', value)


class GroupExpansion(Transformer):

    DEFAULT_ITEM_REPEAT = 4
    DEFAULT_NUM_LINK_PER_NAV_ITEM = 4
    DEFAULT_TOTAL_LINK = 8
    DEFAULT_NAV_TEXT_VALUE = 'Navigation'

    DEFAULT_ELEMENT_REPEAT = 4

    def _get_nav_item_template(self, num_links: int, name: str = None) -> LarkTree:
        if name is None:
            _name = self.DEFAULT_NAV_TEXT_VALUE
        else:
            _name = name
        link_element = create_element('link')
        title_element = create_element('title', [('value', _name)])
        return LarkTree("item", [title_element] + [copy.deepcopy(link_element) for i in range(num_links)])

    def _get_flat_nav_item_template(self, name: str = None, element_type: str = "link") -> LarkTree:
        _name = name or self.DEFAULT_NAV_TEXT_VALUE
        text_element = create_element(element_type, [('value', _name)])
        return LarkTree("item", [text_element])

    def _fill_item_values(self, children, names: List[str], target_element_type: Set,
                          secondary_element_type: Set = None) -> None:
        text_ele_idx, value_attr_idx, name_idx = None, -1, 0
        for _target_element_type in [target_element_type, secondary_element_type]:
            if _target_element_type is None:
                break
            for child in children:
                if name_idx >= len(names):
                    break
                if is_item(child):
                    if text_ele_idx is None:
                        for eidx, e in enumerate(child.children):
                            if is_target_element(e, _target_element_type):
                                text_ele_idx = eidx
                                for aidx, attr in enumerate(e.children):
                                    if is_target_attr(attr, 'value'):
                                        value_attr_idx = aidx
                                        break
                                break
                    if text_ele_idx is None:
                        # No target elements
                        break
                    element = child.children[text_ele_idx]
                    if value_attr_idx >= 0:
                        element.children[value_attr_idx] = create_attr('value', names[name_idx].strip())
                    else:
                        element.children.append(
                            create_attr('value', names[name_idx].strip())
                        )
                    name_idx += 1
            if name_idx > 1:
                break

    def _get_element_types_in_item(self, item: LarkTree) -> Set[str]:
        return {get_element_type(child) for child in item.children if is_element(child)}

    def group(self, children):
        # Handle repeat
        group_type = children[0].value
        total_link, num_repeat, items, names = 0, 0, list(), list()
        for child in children:
            if isinstance(child, LarkTree):
                if child.data == 'item':
                    items.append(child)
                elif child.data == 'group_attr':
                    attr_name, attr_value = child.children[0].value, child.children[1].value.strip("'")
                    if attr_name == 'repeat':
                        num_repeat = int(attr_value) if is_num(attr_value) else self.DEFAULT_ITEM_REPEAT
                    elif attr_name == 'total_link':
                        total_link = int(attr_value) if is_num(attr_value) else self.DEFAULT_TOTAL_LINK
                    elif attr_name == 'names':
                        names = attr_value.split(",")

        if group_type == 'InfoGroup':
            if num_repeat > 0 or len(names) > 0:
                num_repeat = max(num_repeat, len(names))
            if num_repeat > 1 and len(items) == 1:
                children.extend([copy.deepcopy(items[0]) for i in range(num_repeat - 1)])
                self._fill_item_values(children, names, {"title"}, {"link", "button"})
        else:
            # NavGroup & FlatNavGroup
            if num_repeat > 0:
                num_repeat = max(num_repeat, len(names))
                if len(names) < num_repeat:
                    names.extend(
                        [self.DEFAULT_NAV_TEXT_VALUE for i in range(num_repeat - len(names))])
            elif total_link > 0:
                total_link = max(total_link, len(names))
                num_repeat = 0
                for _ in range(max(0, total_link - len(names))):
                    names.append(self.DEFAULT_NAV_TEXT_VALUE)

            if group_type == 'NavGroup':
                # Navigation group in footer
                if total_link == 0 and num_repeat > 0:
                    if len(items) == 1:
                        # item_element_types = self._get_element_types_in_item(items[0])
                        # if len(item_element_types) == 1 and ("link" in item_element_types or "button" in item_element_types):
                        #     children.pop()
                        #     num_link_per_category = self.DEFAULT_NUM_LINK_PER_NAV_ITEM
                        #     for i in range(num_repeat):
                        #         children.append(self._get_nav_item_template(num_link_per_category, names[i].strip()))
                        # else:
                        children.extend([copy.deepcopy(items[0])
                                        for i in range(num_repeat - 1)])
                        self._fill_item_values(children, names, {"title", "text", "button", "link"})
                    elif len(items) == 0:
                        num_link_per_category = self.DEFAULT_NUM_LINK_PER_NAV_ITEM
                        for i in range(num_repeat):
                            children.append(self._get_nav_item_template(num_link_per_category, names[i].strip()))
                elif total_link > 0 and num_repeat > 0:
                    if len(items) == 1:
                        children.extend([copy.deepcopy(items[0])
                                        for i in range(num_repeat - 1)])
                        self._fill_item_values(children, names, {"title", "text", "button", "link"})
                    elif len(items) == 0:
                        num_link_per_category = total_link // num_repeat
                        count = 0
                        for i in range(num_repeat):
                            if i != num_repeat - 1:
                                children.append(self._get_nav_item_template(num_link_per_category, names[i].strip()))
                                count += num_link_per_category
                            else:
                                diff = total_link - count
                                children.append(
                                    self._get_nav_item_template(diff, names[i].strip()))
                elif total_link > 0 and num_repeat == 0:
                    if len(items) == 0:
                        for i in range(total_link):
                            children.append(self._get_flat_nav_item_template(names[i].strip(), element_type="link"))
                    elif len(items) == 1:
                        children.extend([copy.deepcopy(items[0]) for i in range(total_link - 1)])
                        self._fill_item_values(children, names, {"title", "text", "button", "link"})
            else:
                # group_type == 'FlatNavGroup':
                # Navigation group in Header
                if num_repeat > 1 and len(items) == 1:
                    children.extend([copy.deepcopy(items[0])
                                    for i in range(num_repeat - 1)])
                    self._fill_item_values(children, names, TEXT_LIKE_ELEMENTS)
                elif num_repeat > 0 and len(items) == 0:
                    for i in range(num_repeat):
                        children.append(self._get_flat_nav_item_template(names[i].strip()))

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
        "text": "text", "title": "title", "link": "link",
        "description": "description", "image": "image", "background_image": "background",
        "logo": "logo", "icon": "icon", "button": "button", "input": "input",
        "pagination": "pagination"
    }

    POS_PRIORITY = {
        "top": 0, "bottom": 1, "left": 2, "right": 3, "center": 4, "undefined": 5
    }
    POS_MAP = {
        "top": "top", "bottom": "bottom", "left": "left", "right": "right",
        "middle": "center", "top-left": "top", "top-right": "top"
    }

    SIZE_PRIORITY = {
        "small": 0, "large": 1, "undefined": 2
    }

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
        newline, ncol = False, None
        for child in group.children:
            if isinstance(child, LarkTree) and child.data == 'item':
                result.append(self.linearize_item(child))
            elif is_target_attr(child, 'col'):
                ncol = int(get_attr(child).strip("'"))
            elif is_target_attr(child, 'display'):
                newline = get_attr(child).strip("'") == "newline"
        if ncol is None:
            results = [(result, newline,)]
        else:
            results = list()
            beg = 0
            while beg < len(result):
                results.append((result[beg:beg+ncol], newline,))
                beg += ncol
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
        # initialize
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
        output, footer_icons = list(), list()
        for e in elements:
            if region_type == 'footer' and e.etype == "icon":
                footer_icons.append(e)
            else:
                output.append(str(e))
                element_map[e.eid] = num_element
                num_element += 1
        if len(footer_icons) > 0:
            icon_str = list()
            for fe in footer_icons:
                icon_str.append(f"[ {str(fe)} ]")
                element_map[fe.eid] = num_element
                num_element += 1
            icon_str = " | ".join(icon_str)
            output.append("{ " + icon_str + " }")
        for group, newline in groups:
            for item_idx, item in enumerate(group):
                item_output = list()
                for e in item:
                    item_output.append(str(e))
                    element_map[e.eid] = num_element
                    num_element += 1
                output.append("[ {} ]".format(" | ".join(item_output)))
                if item_idx == 0:
                    if region_type == "header":
                        if newline:
                            output[-1] = "{ " + output[-1]
                        else:
                            output[-1] = "{ " + output[-1]
                    else:
                        output[-1] = "{ " + output[-1]
                if item_idx == len(group) - 1:
                    output[-1] = output[-1] + " }"

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
