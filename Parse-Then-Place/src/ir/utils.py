# coding=utf8

from typing import Dict, List, Set, Tuple, Union

from lark import Token
from lark import Tree as LarkTree
from lark.visitors import Transformer


def is_item(subtree: LarkTree) -> bool:
    return isinstance(subtree, LarkTree) and subtree.data == 'item'


def is_group(subtree: LarkTree) -> bool:
    return isinstance(subtree, LarkTree) and subtree.data == 'group'


def is_element(subtree: LarkTree) -> bool:
    return isinstance(subtree, LarkTree) and subtree.data == 'element'


def is_region(subtree: LarkTree) -> bool:
    return isinstance(subtree, LarkTree) and subtree.data == 'region'


def is_target_element(subtree: LarkTree, target: Union[str, Set]) -> bool:
    if not is_element(subtree):
        return False
    if isinstance(target, str):
        return subtree.children[0].value == target
    else:
        # Set
        return subtree.children[0].value in target


def is_attr(subtree: LarkTree) -> bool:
    return isinstance(subtree, LarkTree) and (subtree.data == 'attr' or subtree.data == 'group_attr')


def is_target_attr(subtree: LarkTree, attr_type: str) -> bool:
    if not is_attr(subtree):
        return False
    return subtree.children[0].value == attr_type


def create_attr(attr_type: str, attr_value: str) -> LarkTree:
    return LarkTree('attr', [Token('ATTR_TYPE', attr_type),
                             Token('ATTR_VALUE', f"'{attr_value}'")])


def create_group_attr(attr_type: str, attr_value: str) -> LarkTree:
    return LarkTree('group_attr', [Token('GROUP_ATTR_TYPE', attr_type),
                                   Token('ATTR_VALUE', f"'{attr_value}'")])


def create_element(etype: str, attributes: List[Tuple] = None) -> LarkTree:
    children = [Token('ELEMENT_TYPE', etype)]
    if attributes is not None:
        for attr_type, attr_value in attributes:
            children.append(create_attr(attr_type, attr_value))
    return LarkTree('element', children)


def create_region(rtype: str) -> LarkTree:
    return LarkTree('region', [Token('REGION_TYPE', rtype)])


def create_group(gtype: str, attributes: List[Tuple] = None) -> LarkTree:
    children = [Token('GROUP_TYPE', gtype)]
    if attributes is not None:
        for attr_type, attr_value in attributes:
            children.append(create_group_attr(attr_type, attr_value))
    return LarkTree('group', children)


def create_item() -> LarkTree:
    return LarkTree('item', list())


def get_element_type(element: LarkTree) -> str:
    return element.children[0].value


def get_attr(attr: LarkTree) -> str:
    return attr.children[1].value


def set_attr(attr: LarkTree, value: str) -> None:
    attr.children[1].value = value


def get_attr_type(attr: LarkTree) -> str:
    return attr.children[0].value


def get_group_type(group: LarkTree) -> str:
    return group.children[0].value


def get_region_type(region: LarkTree) -> str:
    return region.children[0].value


class AddElementBoundingBox(Transformer):

    def __init__(self, bbox: List[Tuple[str, Tuple]], emap: Dict[str, int]) -> None:
        super().__init__(visit_tokens=False)
        self.emap = emap
        self.bbox = bbox
        self.is_valid = True
        self.max_bottom = 0

    def element(self, children):
        eid = None
        for child in children:
            if is_target_attr(child, 'eid'):
                eid = get_attr(child).strip("'")
                break

        index = self.emap.get(eid, None)
        if index is None or index >= len(self.bbox):
            self.is_valid = False
            return LarkTree('element', children)

        top, height = self.bbox[index][1][1], self.bbox[index][1][3]
        self.max_bottom = max(self.max_bottom, top+height)
        attr = create_attr('bbox', ",".join(map(str, self.bbox[index][1])))
        children.append(attr)
        return LarkTree('element', children)
