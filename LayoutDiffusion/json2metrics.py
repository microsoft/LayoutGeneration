import io
import sys
import math
import multiprocessing as mp
import pickle
from collections import OrderedDict as OD

import numpy as np
import torch

from eval_src.evaluation.metrics import LayoutFID, compute_maximum_iou, compute_self_sim, compute_alignment, compute_overlap, compute_overlap_ignore_bg
from eval_src.data.transforms import convert_ltwh_to_ltrb, decapulate

try:
    pred_path = sys.argv[1]
except:
    pred_path = "results/generation_outputs/gaussian_refine_pow2.5_aux_lex_ltrb_200_5e5_pub/refine/gaussian_refine_pow2.5_aux_lex_ltrb_200_5e5_pub.ema_0.9999_400000.pt.samples_-1.0_elem1.json"

out_path = "/".join(pred_path.split("/")[0:-1]) + "/processed.pt"

with open(pred_path, "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    words = line.split(" ")
    words_new = [word.strip('["').replace('"]\n', "") for word in words]
    lines[i] = words_new

layouts = []
for line in lines:
    sos = 999
    eos = 0
    tmp = 0
    for i, word in enumerate(line):
        if word == "START":
            sos = i
        elif word == "END":
            eos = i
        # COMMENT BELOW TO SETTING STRICT CONSTRAINED

        elif word == "|":  # last <sep>
            tmp = i
            if sos == 999:  # only set when first time
                sos = i
    if eos == 0 or eos < sos:
        eos = tmp

    # COMMENT ABOVE TO SETTING STRICT CONSTRAINED

    if sos < eos:
        layouts.append(line[sos + 1 : eos])

# separate layouts by type
failed = 0
layouts_sep = []
for list1 in layouts:
    if "|" not in list1:  # only one elem
        LIST = []
        LIST.append(list1[0:5])
        layouts_sep.append(LIST)
        continue

    index_list = []  
    eee = enumerate(list1)
    for index, item in eee:
        if item == "|":
            index_list.append(index)

    index_list = [0] + index_list

    head = list1[0 : index_list[1]]
    tail = list1[index_list[len(index_list) - 1] :]
    tail.pop(0)
    c = len(index_list)
    LIST = []
    i = 1
    for i in range(1, c - 1):
        small_list = []
        small_list = list1[index_list[i] : index_list[i + 1]]
        small_list.pop(0)
        if small_list:
            LIST.append(small_list)
        else:
            failed += 1
        i = i + 1

    LIST.insert(0, head)
    LIST.append(tail)
    layouts_sep.append(LIST)


def continuize(bbox):
    """
    Args:
        discrete_bbox torch.LongTensor: N * 4

    Returns:
        continuous_bbox torch.Tensor: N * 4
    """
    bbox = torch.tensor(bbox).long()
    if "ltwh" not in pred_path:
        # print("no need convert to ltrb")
        x1, y1, x2, y2 = decapulate(bbox)
    else:
        x1, y1, x2, y2 = decapulate(convert_ltwh_to_ltrb(bbox))
    # x1, y1, x2, y2 = decapulate(bbox)
    cx1, cx2 = x1 / 127, x2 / 127
    cy1, cy2 = y1 / 127, y2 / 127
    return torch.stack([cx1, cy1, cx2, cy2], dim=-1).float()


layouts_final = []

labels_RICO = [
    "Text", "Image", "Icon", "List_Item", "Text_Button", "Toolbar", "Web_View", "Input", 
    "Card", "Advertisement", "Background_Image", "Drawer", "Radio_Button",  "Checkbox", 
    "Multi_Tab", "Pager_Indicator", "Modal", "On_Off_Switch", "Slider", "Map_View",
    "Button_Bar", "Video", "Bottom_Navigation", "Number_Stepper", "Date_Picker",
]
labels_Pub = [
    "text", "title", "list", "table", "figure",
]
max = 0
for layout in layouts_sep:
    layout_final = {}
    layout_final["pred"] = []
    bboxs = []
    labels = []
    bboxs_continue = []
    if len(layout) > max:
        max = len(layout)
    for element in layout:
        if not element:
            continue
        if type(bboxs) is not list:
            bboxs = bboxs.tolist()
        if (
            len(element) >= 4
            and str.isdigit(element[-4])
            and str.isdigit(element[-3])
            and str.isdigit(element[-2])
            and str.isdigit(element[-1])
        ):
            bboxs.append([int(num) for num in element[-4:]])
        else:
            continue
        bboxs_continue = continuize(bboxs)

        if "pub" in pred_path:
            if str.isdigit(element[1]):
                if element[0] in labels_Pub:
                    labels.append(labels_Pub.index(element[0]) + 1)
                else:
                    continue
        else:
            if str.isdigit(element[1]):
                if element[0] in labels_RICO:
                    labels.append(labels_RICO.index(element[0]) + 1)
                else:
                    continue
    if len(bboxs_continue) < len(labels):
        labels = labels[: len(bboxs_continue)]
    if len(bboxs_continue) > len(labels):
        bboxs_continue = bboxs_continue[: len(labels)]
    layout_final["pred"].append(bboxs_continue)
    layout_final["pred"].append(labels)
    layouts_final.append(layout_final)

with open(out_path, "wb") as f:
    pickle.dump(layouts_final, f)

if "pub" not in pred_path:
    print(
        "python eval_src/tools/draw_from_results.py -d rico -p "
        + out_path
        + " -s "
        + "/".join(out_path.split("/")[0:-1])
        + "/pics -n 100"
    )
else:
    print(
        "python eval_src/tools/draw_from_results.py -d publaynet -p "
        + out_path
        + " -s "
        + "/".join(out_path.split("/")[0:-1])
        + "/pics -n 100"
    )


pred_path = out_path

if "pub" in pred_path:
    publaynet = 1
else:
    publaynet = 0


def create_fid_model(dataset, device="cpu"):
    if dataset == "rico":
        fid_net_path = "eval_src/net/fid_rico.pth.tar"
        fid_model = LayoutFID(
            max_num_elements=20,
            num_labels=25,  # labels for rico
            net_path=fid_net_path,
            device=device,
        )
        return fid_model
    elif dataset == "publaynet":
        fid_net_path = "eval_src/net/fid_publaynet.pth.tar"
        fid_model = LayoutFID(
            max_num_elements=20,
            num_labels=5,  # labels for
            net_path=fid_net_path,
            device=device,
        )
        return fid_model


# making mask for rico test set
if not publaynet:
    with open("data/raw_datasets/rico/pre_processed_20_25/test.pt", "rb") as f:
        buffer = io.BytesIO(f.read())
        gt = torch.load(buffer)
    gold_bboxes = torch.zeros(len(gt), 20, 4)
    gold_labels = torch.zeros(len(gt), 20).long()
    gold_padding_mask = torch.zeros(len(gt), 20).bool()
    for i, layout in enumerate(gt):
        gold_bboxes[i][0 : len(layout["labels"])] = convert_ltwh_to_ltrb(
            layout["bboxes"]
        )
        gold_labels[i][0 : len(layout["labels"])] = layout["labels"]
        gold_padding_mask[i][0 : len(layout["labels"])] = 1
else:
    with open("data/raw_datasets/publaynet/pre_processed_20_5/test.pt", "rb") as f:
        buffer = io.BytesIO(f.read())
        gt = torch.load(buffer)
    pub_gold_bboxes = torch.zeros(len(gt), 20, 4)
    pub_gold_labels = torch.zeros(len(gt), 20).long()
    pub_gold_padding_mask = torch.zeros(len(gt), 20).bool()
    for i, layout in enumerate(gt):
        pub_gold_bboxes[i][0 : len(layout["labels"])] = convert_ltwh_to_ltrb(
            layout["bboxes"]
        )
        pub_gold_labels[i][0 : len(layout["labels"])] = layout["labels"]
        pub_gold_padding_mask[i][0 : len(layout["labels"])] = 1


if not publaynet:
    with open("data/raw_datasets/rico/pre_processed_20_25/val.pt", "rb") as f:
        buffer = io.BytesIO(f.read())
        val = torch.load(buffer)
    val_bboxes = torch.zeros(len(val), 20, 4)
    val_labels = torch.zeros(len(val), 20).long()
    val_padding_mask = torch.zeros(len(val), 20).bool()
    for i, layout in enumerate(val):
        val_bboxes[i][0 : len(layout["labels"])] = convert_ltwh_to_ltrb(
            layout["bboxes"]
        )
        val_labels[i][0 : len(layout["labels"])] = layout["labels"]
        val_padding_mask[i][0 : len(layout["labels"])] = 1
else:
    with open("data/raw_datasets/publaynet/pre_processed_20_5/val.pt", "rb") as f:
        buffer = io.BytesIO(f.read())
        val = torch.load(buffer)
    pub_val_bboxes = torch.zeros(len(val), 20, 4)
    pub_val_labels = torch.zeros(len(val), 20).long()
    pub_val_padding_mask = torch.zeros(len(val), 20).bool()
    for i, layout in enumerate(val):
        pub_val_bboxes[i][0 : len(layout["labels"])] = convert_ltwh_to_ltrb(
            layout["bboxes"]
        )
        pub_val_labels[i][0 : len(layout["labels"])] = layout["labels"]
        pub_val_padding_mask[i][0 : len(layout["labels"])] = 1


with open(pred_path, "rb") as f:
    results = pickle.load(f)
    print("length of pred set", len(results))
pred_bboxes = torch.zeros(len(results), 20, 4)
pred_labels = torch.zeros(len(results), 20).long()
pred_padding_mask = torch.zeros(len(results), 20).bool()

for i, layout in enumerate(results):
    if type(layout["pred"][0]) is list:
        continue
    pred_bboxes[i][0 : len(layout["pred"][0])] = layout["pred"][0][0:20]
    pred_labels[i][0 : len(layout["pred"][0])] = torch.tensor(layout["pred"][1][0:20])
    pred_padding_mask[i][0 : len(layout["pred"][0])] = 1


if not publaynet:
    overlap_score_new = compute_overlap_ignore_bg(
        pred_bboxes, pred_labels, pred_padding_mask
    )
else:
    overlap_score = compute_overlap(pred_bboxes, pred_padding_mask)

alignment_score = compute_alignment(pred_bboxes, pred_padding_mask)


if publaynet:
    fid_model = create_fid_model("publaynet")
    fid_model.collect_features(
        pub_gold_bboxes.to("cpu"),
        pub_gold_labels.to("cpu"),
        (~pub_gold_padding_mask).to("cpu"),
        real=True,
    )
else:
    fid_model = create_fid_model("rico")
    fid_model.collect_features(
        gold_bboxes.to("cpu"),
        gold_labels.to("cpu"),
        (~gold_padding_mask).to("cpu"),
        real=True,
    )

fid_model.collect_features(
    pred_bboxes.to("cpu"), pred_labels.to("cpu"), (~pred_padding_mask).to("cpu")
)
fid_score_eval = fid_model.compute_score()


class DiscretizeBoundingBox:
    def __init__(self, num_x_grid: int, num_y_grid: int) -> None:
        self.num_x_grid = num_x_grid
        self.num_y_grid = num_y_grid
        self.max_x = self.num_x_grid - 1
        self.max_y = self.num_y_grid - 1

    def discretize(self, bbox):
        cliped_boxes = torch.clip(bbox, min=0.0, max=1.0)
        x1, y1, x2, y2 = decapulate(cliped_boxes)
        discrete_x1 = torch.floor(x1 * self.max_x)
        discrete_y1 = torch.floor(y1 * self.max_y)
        discrete_x2 = torch.floor(x2 * self.max_x)
        discrete_y2 = torch.floor(y2 * self.max_y)
        return torch.stack(
            [discrete_x1, discrete_y1, discrete_x2, discrete_y2], dim=-1
        ).long()

    def discretize_num(self, num: float) -> int:
        return int(math.floor(num * self.max_y))

    def __call__(self, data):
        discrete_bboxes = self.discretize(data)
        bbox = discrete_bboxes
        return bbox

discrete_fn = DiscretizeBoundingBox(128, 128)

def collect_gold_layouts(bboxes, labels, mask, layouts):
    for j in range(labels.size(0)):
        _mask = mask[j]
        box = (discrete_fn(bboxes[j][_mask].cpu())).numpy() / 127
        # print(box)
        label = labels[j][_mask].cpu().numpy()
        layouts.append((box, label))
    return layouts, bboxes

def collect_layouts(bboxes, labels, mask, layouts):
    for j in range(labels.size(0)):
        _mask = mask[j]
        box = bboxes[j][_mask].cpu().numpy()
        label = labels[j][_mask].cpu().numpy()
        layouts.append((box, label))
    return layouts, bboxes

gold_layouts = []
pred_layouts = []

# print(gold_layouts)
if not publaynet:
    gold_layouts, _ = collect_layouts(
        gold_bboxes, gold_labels, gold_padding_mask, gold_layouts
    )
else:
    gold_layouts, _ = collect_layouts(
        pub_gold_bboxes, pub_gold_labels, pub_gold_padding_mask, gold_layouts
    )

pred_layouts, _ = collect_layouts(
    pred_bboxes, pred_labels, pred_padding_mask, pred_layouts
)

miou_value = compute_maximum_iou(gold_layouts, pred_layouts, n_jobs=50)

if publaynet:
    overlap_score_new = overlap_score

print(f"\n miou: {miou_value:.4f}, overlap: {overlap_score_new:.4f}, align: {alignment_score * 100:.4f}, fid: {fid_score_eval:.4f}")