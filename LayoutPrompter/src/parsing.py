import re

import openai
import torch

from utils import CANVAS_SIZE, ID2LABEL


class Parser:
    def __init__(self, dataset: str, output_format: str):
        self.dataset = dataset
        self.output_format = output_format
        self.id2label = ID2LABEL[self.dataset]
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.canvas_width, self.canvas_height = CANVAS_SIZE[self.dataset]

    def _extract_labels_and_bboxes(self, prediction: str):
        if self.output_format == "seq":
            return self._extract_labels_and_bboxes_from_seq(prediction)
        elif self.output_format == "html":
            return self._extract_labels_and_bboxes_from_html(prediction)

    def _extract_labels_and_bboxes_from_html(self, predition: str):
        labels = re.findall('<div class="(.*?)"', predition)[1:]  # remove the canvas
        x = re.findall(r"left:.?(\d+)px", predition)[1:]
        y = re.findall(r"top:.?(\d+)px", predition)[1:]
        w = re.findall(r"width:.?(\d+)px", predition)[1:]
        h = re.findall(r"height:.?(\d+)px", predition)[1:]
        if not (len(labels) == len(x) == len(y) == len(w) == len(h)):
            raise RuntimeError
        labels = torch.tensor([self.label2id[label] for label in labels])
        bboxes = torch.tensor(
            [
                [
                    int(x[i]) / self.canvas_width,
                    int(y[i]) / self.canvas_height,
                    int(w[i]) / self.canvas_width,
                    int(h[i]) / self.canvas_height,
                ]
                for i in range(len(x))
            ]
        )
        return labels, bboxes

    def _extract_labels_and_bboxes_from_seq(self, prediction: str):
        label_set = list(self.label2id.keys())
        seq_pattern = r"(" + "|".join(label_set) + r") (\d+) (\d+) (\d+) (\d+)"
        res = re.findall(seq_pattern, prediction)
        labels = torch.tensor([self.label2id[item[0]] for item in res])
        bboxes = torch.tensor(
            [
                [
                    int(item[1]) / self.canvas_width,
                    int(item[2]) / self.canvas_height,
                    int(item[3]) / self.canvas_width,
                    int(item[4]) / self.canvas_height,
                ]
                for item in res
            ]
        )
        return labels, bboxes

    def __call__(self, predictions):
        if isinstance(predictions, openai.types.completion.Completion):
            predictions = predictions.choices
        if isinstance(predictions[0], openai.types.completion_choice.CompletionChoice):
            predictions = [prediction.text for prediction in predictions]

        parsed_predictions = []
        for prediction in predictions:
            try:
                parsed_predictions.append(self._extract_labels_and_bboxes(prediction))
            except:
                continue
        return parsed_predictions
