import os

import numpy as np
import seaborn as sns
import torch
from PIL import Image, ImageDraw

from utils import CANVAS_SIZE, ID2LABEL, RAW_DATA_PATH


class Visualizer:
    def __init__(self, dataset: str, times: float = 3):
        self.dataset = dataset
        self.times = times
        self.canvas_width, self.canvas_height = CANVAS_SIZE[self.dataset]
        self._colors = None

    def draw_layout(self, labels: torch.Tensor, bboxes: torch.Tensor):
        _canvas_width = self.canvas_width * self.times
        _canvas_height = self.canvas_height * self.times
        img = Image.new("RGB", (_canvas_width, _canvas_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img, "RGBA")
        labels = labels.tolist()
        bboxes = bboxes.tolist()
        areas = [bbox[2] * bbox[3] for bbox in bboxes]
        indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)

        for i in indices:
            bbox, label = bboxes[i], labels[i]
            color = self.colors[label]
            c_fill = color + (100,)
            x1, y1, x2, y2 = bbox
            x2 += x1
            y2 += y1
            x1, x2 = x1 * _canvas_width, x2 * _canvas_width
            y1, y2 = y1 * _canvas_height, y2 * _canvas_height
            draw.rectangle([x1, y1, x2, y2], outline=color, fill=c_fill)
        return img

    @property
    def colors(self):
        if self._colors is None:
            n_colors = len(ID2LABEL[self.dataset]) + 1
            colors = sns.color_palette("husl", n_colors=n_colors)
            self._colors = [tuple(map(lambda x: int(x * 255), c)) for c in colors]
        return self._colors

    def __call__(self, predictions):
        images = []
        for prediction in predictions:
            labels, bboxes = prediction
            img = self.draw_layout(labels, bboxes)
            images.append(img)
        return images


class ContentAwareVisualizer:
    def __init__(self, times: float = 3):
        self.canvas_path = os.path.join(
            RAW_DATA_PATH("posterlayout"), "./test/image_canvas"
        )
        self.canvas_width, self.canvas_height = CANVAS_SIZE["posterlayout"]
        self.canvas_width *= times
        self.canvas_height *= times

    def draw_layout(self, img, elems, elems2):
        drawn_outline = img.copy()
        drawn_fill = img.copy()
        draw_ol = ImageDraw.ImageDraw(drawn_outline)
        draw_f = ImageDraw.ImageDraw(drawn_fill)
        cls_color_dict = {1: "green", 2: "red", 3: "orange"}

        for cls, box in elems:
            if cls[0]:
                draw_ol.rectangle(
                    tuple(box), fill=None, outline=cls_color_dict[cls[0]], width=5
                )

        s_elems = sorted(list(elems2), key=lambda x: x[0], reverse=True)
        for cls, box in s_elems:
            if cls[0]:
                draw_f.rectangle(tuple(box), fill=cls_color_dict[cls[0]])

        drawn_outline = drawn_outline.convert("RGBA")
        drawn_fill = drawn_fill.convert("RGBA")
        drawn_fill.putalpha(int(256 * 0.3))
        drawn = Image.alpha_composite(drawn_outline, drawn_fill)

        return drawn

    def __call__(self, predictions, test_idx):
        images = []
        pic = (
            Image.open(os.path.join(self.canvas_path, f"{test_idx}.png"))
            .convert("RGB")
            .resize((self.canvas_width, self.canvas_height))
        )
        for prediction in predictions:
            labels, bboxes = prediction
            labels = labels.unsqueeze(-1)
            labels = np.array(labels, dtype=int)
            bboxes = np.array(bboxes)
            bboxes[:, 0::2] *= self.canvas_width
            bboxes[:, 1::2] *= self.canvas_height
            bboxes[:, 2] += bboxes[:, 0]
            bboxes[:, 3] += bboxes[:, 1]
            images.append(
                self.draw_layout(pic, zip(labels, bboxes), zip(labels, bboxes))
            )
        return images


def create_image_grid(
    image_list, rows=2, cols=5, border_size=6, border_color=(0, 0, 0)
):
    result_width = (
        image_list[0].width * cols + (cols - 1) * border_size + 2 * border_size
    )
    result_height = (
        image_list[0].height * rows + (rows - 1) * border_size + 2 * border_size
    )
    result_image = Image.new("RGB", (result_width, result_height), border_color)
    draw = ImageDraw.Draw(result_image)

    outer_border_rect = [0, 0, result_width, result_height]
    draw.rectangle(outer_border_rect, outline=border_color, width=border_size)

    for i in range(len(image_list)):
        row = i // cols
        col = i % cols
        x_offset = col * (image_list[i].width + border_size) + border_size
        y_offset = row * (image_list[i].height + border_size) + border_size
        result_image.paste(image_list[i], (x_offset, y_offset))

        if border_size > 0:
            border_rect = [
                x_offset - border_size,
                y_offset - border_size,
                x_offset + image_list[i].width + border_size,
                y_offset + image_list[i].height + border_size,
            ]
            draw.rectangle(border_rect, outline=border_color, width=border_size)

    return result_image
