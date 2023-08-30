# coding=utf8

import copy
import re
from collections import defaultdict, deque
from typing import List, Set, Union

from lark import Token
from lark import Tree as LarkTree

from . import utils
from .parser import FOOTER_REGION, HEADER_REGION, MULTIINFO_REGION


def get_element_types(subtree: LarkTree) -> defaultdict:
    type_count = defaultdict(int)
    for child in subtree.children:
        if utils.is_element(child):
            etype = utils.get_element_type(child)
            type_count[etype] += 1
    return type_count


class Squeeze:

    def __call__(self, pt: LarkTree) -> Union[None, LarkTree]:
        return self._squeeze(copy.deepcopy(pt))

    def _squeeze(self, pt: LarkTree) -> LarkTree:
        raise NotImplementedError()


class SqueezeItem(Squeeze):
    """
    Each item's elements should be the same;
    otherwise, we cannot squeeze the group
    """

    def _get_group(self, pt: LarkTree) -> LarkTree:
        region = pt.children[0]
        group = None
        for child in region.children:
            if utils.is_group(child):
                group = child
                break
        return group

    def is_item_same(self, items: List[LarkTree]) -> bool:
        is_meet = True
        num_elements, element_type_count = None, None
        for item in items:
            if element_type_count is None:
                num_elements = len(item.children)
                element_type_count = get_element_types(item)
            else:
                # check
                if num_elements != len(item.children):
                    is_meet = False
                    break
                for et, et_count in get_element_types(item).items():
                    if element_type_count[et] != et_count:
                        is_meet = False
                        break
                if not is_meet:
                    break
        return is_meet

    def _squeeze(self, pt: LarkTree) -> LarkTree:
        group = self._get_group(pt)
        if group is None:
            return None
        group_items = [child for child in group.children if utils.is_item(child)]

        if not self.is_item_same(group_items):
            return None

        num_repeat = len(group_items)
        template = copy.deepcopy(group_items[0])

        group.children = [Token('GROUP_TYPE', utils.get_group_type(group)),
                          utils.create_group_attr('repeat', num_repeat),
                          template]
        return pt


class SqueezeMultiInfoItems(SqueezeItem):

    def _clean_values(self, item: LarkTree):
        for element in item.children:
            if utils.is_element(element):
                value_attr = None
                for attr in element.children:
                    if utils.is_target_attr(attr, 'value'):
                        value_attr = attr
                        break
                if value_attr is not None:
                    element.children.remove(value_attr)

    def _squeeze(self, pt: LarkTree) -> LarkTree:
        group = self._get_group(pt)
        if group is None:
            return None
        group_items = [child for child in group.children if utils.is_item(child)]

        if not self.is_item_same(group_items):
            return None

        num_repeat = len(group_items)
        template = copy.deepcopy(group_items[0])
        self._clean_values(template)

        group.children = [Token('GROUP_TYPE', utils.get_group_type(group)),
                          utils.create_group_attr('repeat', num_repeat),
                          template]
        return pt


class SqueezeMultiInfoItemsWithNames(SqueezeMultiInfoItems):

    def _get_values(self, items: List[LarkTree]) -> List[str]:
        values = list()
        for item in items:
            for element in item.children:
                if utils.is_target_element(element, 'title'):
                    for attr in element.children:
                        if utils.is_target_attr(attr, 'value'):
                            values.append(utils.get_attr(attr).strip("'"))
        return values

    def _squeeze(self, pt: LarkTree) -> LarkTree:
        group = self._get_group(pt)
        if group is None:
            return None
        group_items = [child for child in group.children if utils.is_item(child)]

        if not self.is_item_same(group_items):
            return None

        names = self._get_values(group_items)
        if len(names) == 0:
            return None

        num_repeat = len(group_items)
        # clean values in the tempalte
        template = copy.deepcopy(group_items[0])
        self._clean_values(template)

        group.children = [Token('GROUP_TYPE', utils.get_group_type(group)),
                          utils.create_group_attr('repeat', num_repeat),
                          utils.create_group_attr('names', ", ".join(names)),
                          template]
        return pt


class SqueezeFooterItemsWithTotalLinks(SqueezeItem):

    def is_item_same(self, group_items) -> bool:
        is_meet = True
        for item in group_items:
            element_type_count = get_element_types(item)
            if not (len(element_type_count) <= 4 and (element_type_count['text'] == 1 or element_type_count['title'] == 1) and \
                element_type_count['link'] >= 1):
                is_meet = False
                break
        return is_meet

    def _squeeze(self, pt: LarkTree) -> LarkTree:
        group = self._get_group(pt)
        if group is None:
            return None
        group_items = [child for child in group.children if utils.is_item(child)]

        if not self.is_item_same(group_items):
            return None

        num_repeat = len(group_items)
        total_links = 0
        for item in group_items:
            for child in item.children:
                if utils.is_target_element(child, 'link'):
                    total_links += 1

        group.children = [Token('GROUP_TYPE', utils.get_group_type(group)),
                          utils.create_group_attr('repeat', num_repeat),
                          utils.create_group_attr('total_links', total_links)]
        return pt


class FuzzySqueezeFooterItems(SqueezeFooterItemsWithTotalLinks):

    def _squeeze(self, pt: LarkTree) -> LarkTree:
        group = self._get_group(pt)
        if group is None:
            return None
        group_items = [child for child in group.children if utils.is_item(child)]

        if not self.is_item_same(group_items):
            return None

        num_repeat = len(group_items)

        group.children = [Token('GROUP_TYPE', utils.get_group_type(group)),
                          utils.create_group_attr('repeat', num_repeat)]
        return pt


class FuzzySqueezeFooterItemsWithNames(SqueezeFooterItemsWithTotalLinks):

    def _get_values(self, items: List[LarkTree]) -> List[str]:
        values = list()
        for item in items:
            for element in item.children:
                if utils.is_target_element(element, 'text') or utils.is_target_element(element, 'title'):
                    for attr in element.children:
                        if utils.is_target_attr(attr, 'value'):
                            values.append(utils.get_attr(attr).strip("'"))
        return values

    def _squeeze(self, pt: LarkTree) -> LarkTree:
        group = self._get_group(pt)
        if group is None:
            return None
        group_items = [child for child in group.children if utils.is_item(child)]

        if not self.is_item_same(group_items):
            return None

        names = self._get_values(group_items)
        if len(names) == 0:
            return None

        num_repeat = len(group_items)
        group.children = [Token('GROUP_TYPE', utils.get_group_type(group)),
                          utils.create_group_attr('repeat', num_repeat),
                          utils.create_group_attr('names', ", ".join(names))]
        return pt


class Synthesizer:
    """
    Synthesize ir from valid placement model inputs
    """

    ELEMENT_TYPES = {
        "text", "image", "background_image", "button", "link",
        "icon", "pagination", "input", "title", "description", "logo"
    }
    UNDEFINED = 'undefined'

    def __init__(self, enable_squeeze: bool = False) -> None:
        if enable_squeeze:
            self.squeeze_fn = {
                MULTIINFO_REGION: [
                    SqueezeMultiInfoItems(),
                    SqueezeMultiInfoItemsWithNames()
                ],
                FOOTER_REGION: [
                    SqueezeItem(),
                    SqueezeFooterItemsWithTotalLinks(),
                    FuzzySqueezeFooterItems(),
                    FuzzySqueezeFooterItemsWithNames()
                ],
                HEADER_REGION: [
                    SqueezeItem()
                ]
            }
        else:
            self.squeeze_fn = None

    def _create_element(self, etype: str, pos: str, size: str, value: str = None,
                        vital_values: Set = None) -> LarkTree:
        attrs = list()
        if pos != self.UNDEFINED:
            attrs.append(('position', pos))
        if size != self.UNDEFINED:
            attrs.append(('size', size))
        if value is not None:
            if value.lower() in vital_values:
                attrs.append(('value', f'{value}'))
            elif etype not in {"description", "image", "background_image", "icon", "logo"}:
                num_words = len(value.split())
                if 1 <= num_words <= 5:
                    attrs.append(('value', f'{value}'))
        return utils.create_element(etype, attrs)

    def _create_group(self, curr_parent: LarkTree, region_type: str) -> LarkTree:
        if curr_parent.data == 'region':
            # group
            if region_type == 'Header':
                group_type = 'FlatNavGroup'
            elif region_type == 'MultiInfo':
                group_type = 'InfoGroup'
            elif region_type == 'Footer':
                group_type = 'NavGroup'
            else:
                raise Exception("SingleInfo should not have any group")
            return utils.create_group(group_type)
        elif curr_parent.data == 'group':
            return utils.create_item()
        else:
            raise Exception(f"Unsupported: {curr_parent.data}")

    def __call__(self, placement_input: str, element_info: List = None, vital_values: Set = None) -> List[LarkTree]:
        pt = list()

        match = re.match(r'(.*?)\s(\[.*\])', placement_input.strip())
        if match is None:
            return pt

        region_type, hierarchy_str = match.group(1), match.group(2)
        region = utils.create_region(region_type)

        hierarchy_str = hierarchy_str.replace("[", " [ ").replace("]", " ] ").strip()
        hierarchy_str = re.sub(r'\s+', " ", hierarchy_str) # remove duplicate space
        tokens = hierarchy_str.split()[1:-1]

        parents = deque()
        parents.append(region)

        tidx, element_id = 0, 0
        while tidx < len(tokens):
            token = tokens[tidx]
            if token in self.ELEMENT_TYPES:
                pos, size = tokens[tidx+1], tokens[tidx+2]
                value = None if (element_info is None or element_id >= len(element_info)) else element_info[element_id]
                element = self._create_element(token, pos, size, value, vital_values)
                parents[-1].children.append(element)
                tidx += 2
                element_id += 1
            elif token == '[':
                # new a Group
                curr_parent = parents[-1]
                new_group = self._create_group(curr_parent, region_type)
                parents.append(new_group)
            elif token == ']':
                # end a group
                curr_group = parents.pop()
                parents[-1].children.append(curr_group)
            tidx += 1

        assert len(parents) == 1
        _pt = LarkTree("start", [parents.pop()])
        pt.append(_pt)

        if self.squeeze_fn:
            for squeeze_fn in self.squeeze_fn.get(region_type, list()):
                alternative = squeeze_fn(_pt)
                if alternative is not None:
                    pt.append(alternative)

        return pt
