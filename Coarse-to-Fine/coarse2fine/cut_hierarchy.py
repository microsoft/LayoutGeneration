from data import transforms

import torch


def convert_ltrb_to_ltwh(bbox):
    bbox = torch.tensor(bbox, dtype=torch.long)
    l, t, r, b = transforms.decapulate(bbox)
    w = r - l
    h = b - t
    return torch.stack([l, t, w, h], axis=-1)


class CutHierarchy():

    def format_layout(self, bboxes, labels):
        box_with_label = list()
        for i in range(len(labels)):
            box_with_label.append((bboxes[i], labels[i]))
        return box_with_label

    def group_layout_to_tree(self, box_with_label, distance_threshold, direction):
        sorted_bbox = sorted(box_with_label, key=lambda x: (x[0][1], x[0][0]))
        sorted_bbox_with_idx = [(sorted_bbox[i][0], i) for i in range(len(sorted_bbox))]
        group_list = self.group_bbox(distance_threshold, sorted_bbox_with_idx, direction=direction,)
        return sorted_bbox, group_list

    def group_bbox(self, distance_threshold, sorted_bbox_with_idx, direction):
        """
        sorted_bbox_with_idx: List: [([x1,y1,x2,y2], idx), [...], ...]
        return List of bbox index in sorted_bbox
        """
        assert len(sorted_bbox_with_idx) > 0, f"bboxes length = {len(sorted_bbox_with_idx)}"
        if len(sorted_bbox_with_idx) == 1:
            return [b[1] for b in sorted_bbox_with_idx]
        root = []
        assert direction in ['x', 'y'], f"direction = {direction}"
        next_direct = 'x' if direction == 'y' else 'y'
        idx_d1 = 1 if direction == 'y' else 0
        idx_d2 = 3 if direction == 'y' else 2

        new_bboxes = sorted(sorted_bbox_with_idx, key=lambda x: x[0][idx_d1])
        # begin_y = new_bboxes[0][0][idx_d1]
        end_y = new_bboxes[0][0][idx_d2]
        begin_idx = 0
        for idx in range(1, len(new_bboxes)):
            if (new_bboxes[idx][0][idx_d1]-distance_threshold) > end_y:
                root.append(self.group_bbox(distance_threshold, new_bboxes[begin_idx:idx], direction=next_direct))
                begin_idx = idx
                end_y = new_bboxes[idx][0][idx_d2]
            else:
                end_y = max(new_bboxes[idx][0][idx_d2], end_y)

        if begin_idx == 0:
            return [b[1] for b in new_bboxes]

        root.append(self.group_bbox(distance_threshold, new_bboxes[begin_idx:], direction=next_direct))
        return root

    def get_cut_structure_bottom_two(self, group_tree, bottom_two_layer=list()):
        flag = True
        for group in group_tree:
            if isinstance(group, list) and len(group) != 1:
                flag = False

        if flag is True:
            if isinstance(group_tree[0], list):
                this_group = []
                for group in group_tree:
                    this_group.append(group[0])
                bottom_two_layer.append(this_group)
            else:
                bottom_two_layer.append(group_tree)
        else:
            for group in group_tree:
                if isinstance(group, list) and (len(group) != 1):
                    bottom_two_layer = self.get_cut_structure_bottom_two(group, bottom_two_layer)
                if isinstance(group, list) and (len(group) == 1):
                    bottom_two_layer.append(group)
                if not isinstance(group, list):
                    bottom_two_layer.append([group])

        return bottom_two_layer

    def get_groupbox_pos(self, structure, sorted_box):
        groupbox_pos = list()
        box_list = [_[0] for _ in sorted_box]
        for group in structure:
            left = 9999
            right = 0
            top = 9999
            bottom = 0
            for i in range(len(group)):
                index = group[i]
                left = min(left, box_list[index][0])
                right = max(right, box_list[index][2])
                top = min(top, box_list[index][1])
                bottom = max(bottom, box_list[index][3])
            groupbox_pos.append([left, top, right, bottom])

        return groupbox_pos

    def get_group_infomation(self, structure, sorted_bbox):
        grouped_label = list()
        grouped_box = list()
        for group in structure:
            label = list()
            box = list()
            for idx in group:
                label.append(sorted_bbox[idx][1])
                box.append(sorted_bbox[idx][0])
            grouped_label.append(torch.tensor(label))
            grouped_box.append(torch.tensor(box))

        return grouped_label, grouped_box

    def get_label_in_one_group(self, grouped_label, num_label):
        label_in_one_group = list()
        for group in grouped_label:
            group_vec = list(0 for i in range(num_label))
            for label in group:
                group_vec[label-1] += 1
            label_in_one_group.append(group_vec)
        return torch.tensor(label_in_one_group)

    def relative_coordinate(self, grouped_box, group_bounding_box):

        for i in range(len(grouped_box)):
            grouped_box[i] = grouped_box[i].to(torch.float64)
            w = max(group_bounding_box[i][2] - group_bounding_box[i][0], 1e-8)
            h = max(group_bounding_box[i][3] - group_bounding_box[i][1], 1e-8)
            for j in range(len(grouped_box[i])):
                grouped_box[i][j][0] = (grouped_box[i][j][0] - group_bounding_box[i][0]) / w
                grouped_box[i][j][1] = (grouped_box[i][j][1] - group_bounding_box[i][1]) / h
                grouped_box[i][j][2] = (grouped_box[i][j][2] - group_bounding_box[i][0]) / w
                grouped_box[i][j][3] = (grouped_box[i][j][3] - group_bounding_box[i][1]) / h

        return grouped_box

    def __call__(self, data, num_labels, discrete_func):

        nd = dict()
        bboxes = transforms.convert_ltwh_to_ltrb(data['discrete_gold_bboxes']).tolist()
        labels = data['labels'].tolist()
        box_with_label = self.format_layout(bboxes, labels)

        sorted_bbox, group_tree = self.group_layout_to_tree(box_with_label, distance_threshold=0, direction='y')
        if len(group_tree) == len(sorted_bbox):
            sorted_bbox, group_tree = self.group_layout_to_tree(box_with_label, distance_threshold=0, direction='x')

        structure = self.get_cut_structure_bottom_two(group_tree, list())
        group_bounding_box = self.get_groupbox_pos(structure, sorted_bbox)

        grouped_label, grouped_box = self.get_group_infomation(structure, sorted_bbox)
        relative_grouped_box = self.relative_coordinate(grouped_box, group_bounding_box)

        nd["group_bounding_box"] = convert_ltrb_to_ltwh(group_bounding_box)
        nd["label_in_one_group"] = self.get_label_in_one_group(grouped_label, num_labels)

        nd["grouped_label"] = grouped_label
        nd["grouped_box"] = [convert_ltrb_to_ltwh(discrete_func(group)) for group in relative_grouped_box]

        return nd
