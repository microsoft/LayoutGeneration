# coding=utf8

import copy
from typing import List, Dict, Set, Union, Tuple

import torch

from data.transforms import DiscretizeBoundingBox
from model.layout_transformer import LayoutTransformerTokenizer


class TransformerSortByDictConstraintDecodeState:

    ELEMENT = 'element'
    NUMBER = 'number'
    SEP = 'sep'

    def __init__(self, num_elements: int) -> None:
        self.num_elements = num_elements
        self.curr_element = 0
        self.next_token_type = self.ELEMENT
        self.num_bbox = 0
        self.pred_labels = list()
        self.pred_bbox = list()

    @property
    def finished(self):
        return self.curr_element == self.num_elements and self.num_bbox >= 4

    def add_label(self, label_token: str):
        self.pred_labels.append(label_token)
        self.pred_bbox.append(list())

    def add_bbox_num(self, num: str):
        self.pred_bbox[-1].append(int(num))


class TransformerSortByDictLabelConstraint:

    def __init__(self, tokenizer: LayoutTransformerTokenizer,
                 discrete_degree: int, label_set: Union[List, Set], index2label: Dict,
                 add_sep_token: bool = False, sep_token: str = '|') -> None:
        self.tokenizer = tokenizer
        self.index2label = index2label
        self.add_sep_token = add_sep_token
        self.sep_token = sep_token
        self.special_token_ids = {
            tokenizer.pad_token_id, tokenizer.eos_token_id}

        self.nums = set(range(discrete_degree))
        self.num_tokens = list(map(str, self.nums))
        self.num_token_ids = set(
            self.tokenizer.convert_tokens_to_ids(self.num_tokens))

        self.label_token_ids = dict()
        self.label_suffix = set()
        for label in label_set:
            self.label_token_ids[label.lower()] = set(
                self.tokenizer.convert_tokens_to_ids([label.lower()]))
            self.label_suffix |= self.label_token_ids[label.lower()]

        if self.add_sep_token:
            # NOTE: be careful, the type of label suffix depends on the sep token
            token_ids = set(self.tokenizer.convert_tokens_to_ids([sep_token]))
            self.label_suffix = token_ids

    @property
    def all_plausible_tokens(self):
        token_ids = self.special_token_ids | self.num_token_ids
        for label in self.label_token_ids.keys():
            token_ids |= self.label_token_ids[label]
        return token_ids

    def prepare(self, label_ids: List[List[int]]):
        self.constraints, self.decode_state = list(), list()
        for item_label_ids in label_ids:
            # lid 0 are pad labels
            if isinstance(item_label_ids, list):
                _item_label_ids = item_label_ids
            else:
                _item_label_ids = item_label_ids.tolist()
            item_label_names = [self.index2label[lid].lower().strip()
                                for lid in _item_label_ids if lid > 0]
            item_label_token_id = self.tokenizer.convert_tokens_to_ids(
                item_label_names)
            self.constraints.append(item_label_token_id)
            self.decode_state.append(
                [TransformerSortByDictConstraintDecodeState(len(item_label_names))])

    def __call__(self, batch_id: int, seq_id: int, token_ids: torch.Tensor) -> List[int]:

        label_constraints = self.constraints[batch_id]
        self.decode_state[batch_id] = self.decode_state[batch_id][:seq_id+1]
        state = copy.deepcopy(self.decode_state[batch_id][-1])

        if state.finished:
            plausible_token_id = self.special_token_ids
        else:
            if state.next_token_type == TransformerSortByDictConstraintDecodeState.ELEMENT:
                plausible_token_id = {label_constraints[state.curr_element]}
                state.curr_element += 1
                state.next_token_type = TransformerSortByDictConstraintDecodeState.NUMBER
                state.num_bbox = 0
            elif state.next_token_type == TransformerSortByDictConstraintDecodeState.NUMBER:
                plausible_token_id = self.num_token_ids
                state.num_bbox += 1
                if state.num_bbox >= 4:
                    if self.add_sep_token:
                        state.next_token_type = TransformerSortByDictConstraintDecodeState.SEP
                    else:
                        state.next_token_type = TransformerSortByDictConstraintDecodeState.ELEMENT
            else:
                # SEP
                plausible_token_id = self.label_suffix
                state.next_token_type = TransformerSortByDictConstraintDecodeState.ELEMENT

            self.decode_state[batch_id].append(state)

        return list(plausible_token_id), None


class TransformerSortByDictLabelSizeConstraint(TransformerSortByDictLabelConstraint):

    def prepare(self, label_ids: List[List[int]], bboxes: List[List[int]]):
        self.label_constraints, self.size_constraints, self.decode_state = list(), list(), list()
        for item_label_ids, item_bboxes in zip(label_ids, bboxes):
            # lid 0 are pad labels
            if isinstance(item_label_ids, list):
                _item_label_ids = item_label_ids
            else:
                _item_label_ids = item_label_ids.tolist()
            item_label_names = [self.index2label[lid].lower().strip()
                                for lid in _item_label_ids if lid > 0]
            item_label_token_id = self.tokenizer.convert_tokens_to_ids(
                item_label_names)

            if isinstance(item_bboxes, list):
                _item_bboxes = item_bboxes
            else:
                _item_bboxes = item_bboxes.tolist()
            item_size_token_id = [self.tokenizer.convert_tokens_to_ids(list(map(str, bbox[2:])))
                                  for bbox in _item_bboxes]

            self.label_constraints.append(item_label_token_id)
            self.size_constraints.append(item_size_token_id)
            self.decode_state.append(
                [TransformerSortByDictConstraintDecodeState(len(item_label_names))])

    def __call__(self, batch_id: int, seq_id: int, token_ids: torch.Tensor) -> List[int]:

        label_constraints = self.label_constraints[batch_id]
        size_constraints = self.size_constraints[batch_id]
        self.decode_state[batch_id] = self.decode_state[batch_id][:seq_id+1]
        state = copy.deepcopy(self.decode_state[batch_id][-1])

        if state.finished:
            plausible_token_id = self.special_token_ids
        else:
            if state.next_token_type == TransformerSortByDictConstraintDecodeState.ELEMENT:
                plausible_token_id = {label_constraints[state.curr_element]}
                state.curr_element += 1
                state.next_token_type = TransformerSortByDictConstraintDecodeState.NUMBER
                state.num_bbox = 0
            elif state.next_token_type == TransformerSortByDictConstraintDecodeState.NUMBER:
                # size
                if state.num_bbox >= 2:
                    plausible_token_id = {
                        size_constraints[state.curr_element-1][state.num_bbox - 2]}
                else:
                    plausible_token_id = self.num_token_ids
                state.num_bbox += 1
                if state.num_bbox >= 4:
                    if self.add_sep_token:
                        state.next_token_type = TransformerSortByDictConstraintDecodeState.SEP
                    else:
                        state.next_token_type = TransformerSortByDictConstraintDecodeState.ELEMENT
            else:
                # SEP
                plausible_token_id = self.label_suffix
                state.next_token_type = TransformerSortByDictConstraintDecodeState.ELEMENT
            self.decode_state[batch_id].append(state)

        return list(plausible_token_id), None


class TransformerSortByDictRelationConstraint(TransformerSortByDictLabelConstraint):

    def __init__(self, tokenizer: LayoutTransformerTokenizer, discrete_degree: int,
                 label_set: Union[List, Set], index2label: Dict, rel_index2type: Dict,
                 add_sep_token: bool = False, sep_token: str = '|',) -> None:
        super().__init__(tokenizer, discrete_degree, label_set, index2label,
                         add_sep_token, sep_token)
        self.rel_index2type = rel_index2type
        self.discrete_degree = discrete_degree
        self.discrete_fn = DiscretizeBoundingBox(
            discrete_degree, discrete_degree)

    def prepare(self, label_ids: List[List[int]], relations: List[Tuple]) -> None:
        self.label_constraints, self.relation_constraints, self.decode_state = list(), list(), list()

        for item_label_ids, item_relations in zip(label_ids, relations):

            # lid 0 are pad labels
            if isinstance(item_label_ids, list):
                _item_label_ids = item_label_ids
            else:
                _item_label_ids = item_label_ids.tolist()

            item_label_names, label_index_map = list(), dict()
            for lid in _item_label_ids:
                if lid <= 0:
                    continue
                item_label_names.append(self.index2label[lid].lower().strip())
                if lid not in label_index_map:
                    label_index_map[lid] = len(item_label_names) - 1

            item_label_token_id = self.tokenizer.convert_tokens_to_ids(
                item_label_names)

            self.label_constraints.append(item_label_token_id)
            self.decode_state.append(
                [TransformerSortByDictConstraintDecodeState(len(item_label_names))])

            # relations
            item_ele_rel_constraints = [list()
                                        for _ in range(len(item_label_names))]
            if isinstance(item_relations, List):
                _item_relations = item_relations
            else:
                _item_relations = item_relations.tolist()
            for rel in _item_relations:
                label_j, index_j, label_i, index_i, rel_type_idx = rel
                rel_type = self.rel_index2type[rel_type_idx]

                if label_i == 0:
                    raise Exception("label i should not be canvas")
                label_i_pos = label_index_map[label_i] + index_i - 1

                is_canvas = label_j == 0
                if is_canvas:
                    item_ele_rel_constraints[label_i_pos].append(
                        (f"canvas_{rel_type}", None,))
                else:
                    label_j_pos = label_index_map[label_j] + index_j - 1
                    item_ele_rel_constraints[label_i_pos].append(
                        (rel_type, label_j_pos))
                    if rel_type in {"top", "bottom"}:
                        item_ele_rel_constraints[label_j_pos].append(
                            (f"being_{rel_type}", None))

            self.relation_constraints.append(item_ele_rel_constraints)

    def _intersect(self, a: Set, b: Set) -> Set:
        if len(b) == 0:
            return a
        intersection = a & b
        return intersection
        # if len(intersection) > 0:
        #     return intersection
        # return a

    def __call__(self, batch_id: int, seq_id: int, token_ids: torch.Tensor) -> List[int]:

        label_constraints = self.label_constraints[batch_id]
        self.decode_state[batch_id] = self.decode_state[batch_id][:seq_id+1]
        state = copy.deepcopy(self.decode_state[batch_id][-1])

        if len(token_ids) > 0:
            curr_token_id = token_ids[-1].item()
            token = self.tokenizer.convert_ids_to_tokens([curr_token_id])[0]
            if token in self.label_token_ids:
                state.add_label(token)
            elif token in self.num_tokens:
                state.add_bbox_num(token)

        back_idx = None
        if state.finished:
            plausible_token_id = self.special_token_ids
        else:
            if state.next_token_type == TransformerSortByDictConstraintDecodeState.ELEMENT:
                plausible_token_id = {label_constraints[state.curr_element]}
                state.curr_element += 1
                state.next_token_type = TransformerSortByDictConstraintDecodeState.NUMBER
                state.num_bbox = 0
            elif state.next_token_type == TransformerSortByDictConstraintDecodeState.NUMBER:
                relation_constraints = self.relation_constraints[batch_id][state.curr_element-1]
                if len(relation_constraints) == 0:
                    plausible_token_id = self.num_token_ids
                else:
                    plausible_tokens = copy.deepcopy(self.nums)
                    if state.num_bbox < 2:
                        # pos
                        if state.num_bbox == 0:
                            # x
                            for rel_type, tgt_ele_idx in relation_constraints:
                                if tgt_ele_idx is None:
                                    tgt_ele_bbox = [
                                        0, 0, self.discrete_degree - 1, self.discrete_degree - 1]
                                else:
                                    tgt_ele_bbox = state.pred_bbox[tgt_ele_idx]
                                    # if back_idx == None:
                                    #     back_idx = tgt_ele_idx * 5 + 1
                                    back_idx = tgt_ele_idx * 5 + 1
                                if rel_type == "left":
                                    tgt_ele_left = tgt_ele_bbox[0]
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, set(range(tgt_ele_left)))
                                elif rel_type == "right":
                                    tgt_ele_right = tgt_ele_bbox[0] + \
                                        tgt_ele_bbox[2]
                                    plausible_tokens = self._intersect(plausible_tokens, set(
                                        range(tgt_ele_right, self.discrete_degree)))
                                elif rel_type == "center":
                                    tgt_ele_right = tgt_ele_bbox[0] + \
                                        tgt_ele_bbox[2]
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, set(range(tgt_ele_right)))
                                elif rel_type == 'canvas_equal':
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, {0})
                        else:
                            # y
                            for rel_type, tgt_ele_idx in relation_constraints:
                                if tgt_ele_idx is None:
                                    tgt_ele_bbox = [
                                        0, 0, self.discrete_degree - 1, self.discrete_degree - 1]
                                else:
                                    tgt_ele_bbox = state.pred_bbox[tgt_ele_idx]
                                    # if back_idx == None:
                                    #     back_idx = tgt_ele_idx * 5 + 2
                                    back_idx = tgt_ele_idx * 5 + 2
                                if rel_type == "top":
                                    tgt_ele_top = max(tgt_ele_bbox[1] - 1, 1)
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, set(range(tgt_ele_top)))
                                elif rel_type == 'bottom':
                                    tgt_ele_bottom = min(
                                        tgt_ele_bbox[1] + tgt_ele_bbox[3] + 3, self.discrete_degree-1)
                                    plausible_tokens = self._intersect(plausible_tokens, set(
                                        range(tgt_ele_bottom, self.discrete_degree)))
                                elif rel_type in {"left", "right", "center"}:
                                    tgt_ele_bottom = tgt_ele_bbox[1] + \
                                        tgt_ele_bbox[3]
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, set(range(tgt_ele_bottom+1)))
                                elif rel_type == 'canvas_equal':
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, {0})
                                elif rel_type == 'canvas_top':
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, set(range(self.discrete_degree // 3 + 1)))
                                elif rel_type == 'canvas_bottom':
                                    plausible_tokens = self._intersect(plausible_tokens, set(
                                        range(self.discrete_degree // 3, self.discrete_degree)))
                                elif rel_type == 'canvas_center':
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, set(range(2 * self.discrete_degree // 3)))
                                elif rel_type == 'being_top':
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, set(range(5, self.discrete_degree)))
                    else:
                        # size
                        if state.num_bbox == 2:
                            # width
                            curr_x, curr_y = state.pred_bbox[-1]

                            for rel_type, tgt_ele_idx in relation_constraints:
                                if tgt_ele_idx is None:
                                    tgt_ele_bbox = [
                                        0, 0, self.discrete_degree - 1, self.discrete_degree - 1]
                                else:
                                    tgt_ele_bbox = state.pred_bbox[tgt_ele_idx]
                                    # if back_idx == None:
                                    #     back_idx = tgt_ele_idx * 5 + 3
                                    back_idx = tgt_ele_idx * 5 + 3
                                if rel_type == "left":
                                    tgt_ele_left = tgt_ele_bbox[0]
                                    diff = max(tgt_ele_left-curr_x+1, 1)
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, set(range(diff)))
                                elif rel_type == "right":
                                    diff = max(self.discrete_degree-curr_x, 1)
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, set(range(diff)))
                                elif rel_type == "center":
                                    tgt_ele_left = tgt_ele_bbox[0]
                                    if tgt_ele_left > curr_x:
                                        min_width = tgt_ele_left - curr_x
                                        max_width = self.discrete_degree - curr_x
                                        plausible_tokens = self._intersect(
                                            plausible_tokens, set(range(min_width, max_width)))
                                elif rel_type == "smaller":
                                    tgt_ele_size = tgt_ele_bbox[2] * \
                                        tgt_ele_bbox[3]
                                    diff = min(
                                        tgt_ele_size, self.discrete_degree)
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, set(range(diff)))
                                elif rel_type == "larger":
                                    continual_tgt_ele_size = self.discrete_fn.continuize_num(
                                        tgt_ele_bbox[2] * tgt_ele_bbox[3])
                                    continual_tgt_target_size = continual_tgt_ele_size * 1.1
                                    _tgt_target_size = self.discrete_fn.discretize_num(
                                        continual_tgt_target_size)
                                    min_width = max(
                                        0, _tgt_target_size // (self.discrete_degree - 1))
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, set(range(min_width, self.discrete_degree)))
                                elif rel_type == "canvas_equal":
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, set(self.discrete_degree - 1))
                                elif rel_type == "equal":
                                    continual_tgt_ele_size = self.discrete_fn.continuize_num(
                                        tgt_ele_bbox[2] * tgt_ele_bbox[3])
                                    _tgt_target_size = self.discrete_fn.discretize_num(
                                        continual_tgt_ele_size)
                                    diff = min(_tgt_target_size,
                                               self.discrete_degree-1)
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, set(range(diff+1)))
                        else:
                            # height
                            curr_x, curr_y, curr_width = state.pred_bbox[-1]
                            for rel_type, tgt_ele_idx in relation_constraints:
                                if tgt_ele_idx is None:
                                    tgt_ele_bbox = [
                                        0, 0, self.discrete_degree - 1, self.discrete_degree - 1]
                                else:
                                    tgt_ele_bbox = state.pred_bbox[tgt_ele_idx]
                                    # if back_idx == None:
                                    #     back_idx = tgt_ele_idx * 5 + 4
                                    back_idx = tgt_ele_idx * 5 + 4
                                if rel_type in {"left", "right", "center"}:
                                    tgt_ele_top = tgt_ele_bbox[1]
                                    if curr_y < tgt_ele_top:
                                        diff = tgt_ele_top - curr_y
                                        plausible_tokens = self._intersect(
                                            plausible_tokens, set(range(diff, self.discrete_degree)))
                                elif rel_type == "top":
                                    tgt_ele_top = tgt_ele_bbox[1]
                                    diff = max(tgt_ele_top - curr_y - 1, 1)
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, set(range(diff)))
                                elif rel_type == "bottom":
                                    diff = max(self.discrete_degree-curr_y, 1)
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, set(range(diff)))
                                elif rel_type == "smaller":
                                    tgt_ele_size = tgt_ele_bbox[2] * \
                                        tgt_ele_bbox[3]
                                    _curr_width = max(1, curr_width)
                                    max_height = tgt_ele_size // _curr_width
                                    _range = max(
                                        min(max_height - 3, self.discrete_degree), 1)
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, set(range(_range)))
                                elif rel_type == "larger":
                                    continual_tgt_ele_size = self.discrete_fn.continuize_num(
                                        tgt_ele_bbox[2] * tgt_ele_bbox[3])
                                    continual_tgt_target_size = continual_tgt_ele_size * 1.1
                                    _tgt_target_size = self.discrete_fn.discretize_num(
                                        continual_tgt_target_size)
                                    _curr_width = max(1, curr_width)
                                    min_height = _tgt_target_size // _curr_width + 3
                                    min_height = min(
                                        min_height, self.discrete_degree)
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, set(range(min_height, self.discrete_degree)))
                                elif rel_type == "equal":
                                    continual_tgt_ele_size = self.discrete_fn.continuize_num(
                                        tgt_ele_bbox[2] * tgt_ele_bbox[3])
                                    _tgt_target_size = self.discrete_fn.discretize_num(
                                        continual_tgt_ele_size)
                                    _curr_width = max(1, curr_width)
                                    height = _tgt_target_size // _curr_width
                                    height = min(
                                        height, self.discrete_degree-1)
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, {height, max(0, height-1), height+1})
                                elif rel_type == "canvas_equal":
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, set(self.discrete_degree - 1))
                                elif rel_type == "canvas_top":
                                    max_height = max(
                                        1, 2 * (self.discrete_degree // 3 - curr_y))
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, set(range(max_height)))
                                elif rel_type == "canvas_bottom":
                                    min_height = min(
                                        2 * (2 * self.discrete_degree // 3 - curr_y), self.discrete_degree)
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, set(range(min_height, self.discrete_degree)))
                                elif rel_type == "canvas_center":
                                    f, l = self.discrete_degree // 3, 2 * self.discrete_degree // 3
                                    if curr_y < f:
                                        min_height = 2 * (f - curr_y)
                                        max_height = 2 * (l - curr_y)
                                        plausible_tokens = self._intersect(
                                            plausible_tokens, set(range(min_height, max_height+1)))
                                    elif f <= curr_y < l:
                                        max_height = 2 * (l - curr_y)
                                        plausible_tokens = self._intersect(
                                            plausible_tokens, set(range(max_height+1)))
                                elif rel_type == 'being_bottom':
                                    diff = max(
                                        self.discrete_degree - 1 - curr_y, 1)
                                    plausible_tokens = self._intersect(
                                        plausible_tokens, set(range(diff)))

                    if len(plausible_tokens) > 0:
                        min_value, max_value = min(
                            plausible_tokens), max(plausible_tokens)
                        plausible_tokens |= {i for i in range(
                            min_value-2, min_value) if i > 0}
                        plausible_tokens |= {i for i in range(
                            max_value+1, max_value+3) if i < self.discrete_degree}
                        plausible_token_id = self.tokenizer.convert_tokens_to_ids(
                            list(map(str, plausible_tokens)))
                        back_idx = None
                    else:
                        # print("Conflicts")
                        plausible_token_id = self.num_token_ids
                state.num_bbox += 1
                if state.num_bbox >= 4:
                    if self.add_sep_token:
                        state.next_token_type = TransformerSortByDictConstraintDecodeState.SEP
                    else:
                        state.next_token_type = TransformerSortByDictConstraintDecodeState.ELEMENT
            else:
                # SEP
                plausible_token_id = self.label_suffix
                state.next_token_type = TransformerSortByDictConstraintDecodeState.ELEMENT
            self.decode_state[batch_id].append(state)
        return list(plausible_token_id), back_idx
