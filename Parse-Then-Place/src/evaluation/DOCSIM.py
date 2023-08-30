from math import sqrt

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance


def get_ele_size(box):
    # w h
    return [box[2], box[3]]


def get_ele_center(box):
    # xc yc
    return [box[0] + box[2]/2.0, box[1] + box[3]/2.0]


def get_shape_diff(box_a, box_b):
    '''
    |w_a - w_b| + |h_a - h_b|
    '''
    [width_a, height_a] = get_ele_size(box_a)
    [width_b, height_b] = get_ele_size(box_b)
    return abs(width_b-width_a) + abs(height_b-height_a)


def get_pos_diff(box_a, box_b):
    '''
    (center_a - center_b)^2
    '''
    tmp_center_a = get_ele_center(box_a)
    tmp_center_b = get_ele_center(box_b)
    center_a = [tmp_center_a[0], tmp_center_a[1]]
    center_b = [tmp_center_b[0], tmp_center_b[1]]
    return distance.euclidean(center_a, center_b)


def get_area_factor(box_a, box_b):
    '''
    sqrt(min(x_w*x_h, y_w*y_h))
    '''
    [width_a, height_a] = get_ele_size(box_a)
    [width_b, height_b] = get_ele_size(box_b)
    return sqrt(min(width_a*height_a, width_b*height_b))


def get_ele_sim(box_a, label_a, box_b, label_b):
    if label_a != label_b:
        return 0
    pos_diff = get_pos_diff(box_a, box_b)
    shape_diff = get_shape_diff(box_a, box_b)
    area_factor = get_area_factor(box_a, box_b)
    return area_factor * (pow(2, -pos_diff-2*shape_diff))


def get_layout_sim(boxes_a, labels_a, boxes_b, labels_b):
    # get similarity between arbitrary two boxes in two layouts
    ele_sim = []
    for box_a, label_a in zip(boxes_a, labels_a):
        tmp_ele_sim = []
        for box_b, label_b in zip(boxes_b, labels_b):
            tmp_ele_sim.append(get_ele_sim(box_a, label_a, box_b, label_b))
        ele_sim.append(tmp_ele_sim)

    # maximum weight matching
    cost_matrix = np.array(ele_sim)
    row_ind, col_ind = linear_sum_assignment(cost_matrix, True)

    return cost_matrix[row_ind, col_ind].sum()#/cost_matrix[row_ind, col_ind].size