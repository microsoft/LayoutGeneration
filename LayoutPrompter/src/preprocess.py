import copy
import random

import cv2
import torch
import torchvision.transforms as T
from pandas import DataFrame

from transforms import (
    AddCanvasElement,
    AddGaussianNoise,
    AddRelation,
    CLIPTextEncoder,
    DiscretizeBoundingBox,
    LabelDictSort,
    LexicographicSort,
    SaliencyMapToBBoxes,
    ShuffleElements,
)
from utils import CANVAS_SIZE, ID2LABEL, clean_text


class Processor:
    def __init__(
        self, index2label: dict, canvas_width: int, canvas_height: int, *args, **kwargs
    ):
        self.index2label = index2label
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.sort_by_pos = kwargs.get("sort_by_pos", None)
        self.shuffle_before_sort_by_label = kwargs.get(
            "shuffle_before_sort_by_label", None
        )
        self.sort_by_pos_before_sort_by_label = kwargs.get(
            "sort_by_pos_before_sort_by_label", None
        )

        if not any(
            [
                self.sort_by_pos,
                self.shuffle_before_sort_by_label,
                self.sort_by_pos_before_sort_by_label,
            ]
        ):
            raise ValueError(
                "At least one of sort_by_pos, shuffle_before_sort_by_label, or sort_by_pos_before_sort_by_label must be True."
            )
        self.transform_functions = self._config_base_transform()

    def _config_base_transform(self):
        transform_functions = list()
        if self.sort_by_pos:
            transform_functions.append(LexicographicSort())
        else:
            if self.shuffle_before_sort_by_label:
                transform_functions.append(ShuffleElements())
            elif self.sort_by_pos_before_sort_by_label:
                transform_functions.append(LexicographicSort())
            transform_functions.append(LabelDictSort(self.index2label))
        transform_functions.append(
            DiscretizeBoundingBox(
                num_x_grid=self.canvas_width, num_y_grid=self.canvas_height
            )
        )
        return transform_functions

    def __call__(self, data):
        _data = self.transform(copy.deepcopy(data))
        return {k: _data[k] for k in self.return_keys}


class GenTypeProcessor(Processor):
    return_keys = [
        "labels",
        "bboxes",
        "gold_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
    ]

    def __init__(
        self,
        index2label: dict,
        canvas_width: int,
        canvas_height: int,
        sort_by_pos: bool = False,
        shuffle_before_sort_by_label: bool = False,
        sort_by_pos_before_sort_by_label: bool = True,
    ):
        super().__init__(
            index2label=index2label,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            sort_by_pos=sort_by_pos,
            shuffle_before_sort_by_label=shuffle_before_sort_by_label,
            sort_by_pos_before_sort_by_label=sort_by_pos_before_sort_by_label,
        )
        self.transform = T.Compose(self.transform_functions)


class GenTypeSizeProcessor(Processor):
    return_keys = [
        "labels",
        "bboxes",
        "gold_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
    ]

    def __init__(
        self,
        index2label: dict,
        canvas_width: int,
        canvas_height: int,
        sort_by_pos: bool = False,
        shuffle_before_sort_by_label: bool = True,
        sort_by_pos_before_sort_by_label: bool = False,
    ):
        super().__init__(
            index2label=index2label,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            sort_by_pos=sort_by_pos,
            shuffle_before_sort_by_label=shuffle_before_sort_by_label,
            sort_by_pos_before_sort_by_label=sort_by_pos_before_sort_by_label,
        )
        self.transform = T.Compose(self.transform_functions)


class GenRelationProcessor(Processor):
    return_keys = [
        "labels",
        "bboxes",
        "gold_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
        "relations",
    ]

    def __init__(
        self,
        index2label: dict,
        canvas_width: int,
        canvas_height: int,
        sort_by_pos: bool = False,
        shuffle_before_sort_by_label: bool = False,
        sort_by_pos_before_sort_by_label: bool = True,
        relation_constrained_discrete_before_induce_relations: bool = False,
    ):
        super().__init__(
            index2label=index2label,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            sort_by_pos=sort_by_pos,
            shuffle_before_sort_by_label=shuffle_before_sort_by_label,
            sort_by_pos_before_sort_by_label=sort_by_pos_before_sort_by_label,
        )
        self.transform_functions = self.transform_functions[:-1]
        if relation_constrained_discrete_before_induce_relations:
            self.transform_functions.append(
                DiscretizeBoundingBox(
                    num_x_grid=self.canvas_width, num_y_grid=self.canvas_height
                )
            )
            self.transform_functions.append(
                AddCanvasElement(
                    use_discrete=True, discrete_fn=self.transform_functions[-1]
                )
            )
            self.transform_functions.append(AddRelation())
        else:
            self.transform_functions.append(AddCanvasElement())
            self.transform_functions.append(AddRelation())
            self.transform_functions.append(
                DiscretizeBoundingBox(
                    num_x_grid=self.canvas_width, num_y_grid=self.canvas_height
                )
            )
        self.transform = T.Compose(self.transform_functions)


class CompletionProcessor(Processor):
    return_keys = [
        "labels",
        "bboxes",
        "gold_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
    ]

    def __init__(
        self,
        index2label: dict,
        canvas_width: int,
        canvas_height: int,
        sort_by_pos: bool = True,
        shuffle_before_sort_by_label: bool = False,
        sort_by_pos_before_sort_by_label: bool = False,
    ):
        super().__init__(
            index2label=index2label,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            sort_by_pos=sort_by_pos,
            shuffle_before_sort_by_label=shuffle_before_sort_by_label,
            sort_by_pos_before_sort_by_label=sort_by_pos_before_sort_by_label,
        )
        self.transform = T.Compose(self.transform_functions)


class RefinementProcessor(Processor):
    return_keys = [
        "labels",
        "bboxes",
        "gold_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
    ]

    def __init__(
        self,
        index2label: dict,
        canvas_width: int,
        canvas_height: int,
        sort_by_pos: bool = False,
        shuffle_before_sort_by_label: bool = False,
        sort_by_pos_before_sort_by_label: bool = True,
        gaussian_noise_mean: float = 0.0,
        gaussian_noise_std: float = 0.01,
        train_bernoulli_beta: float = 1.0,
    ):
        super().__init__(
            index2label=index2label,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            sort_by_pos=sort_by_pos,
            shuffle_before_sort_by_label=shuffle_before_sort_by_label,
            sort_by_pos_before_sort_by_label=sort_by_pos_before_sort_by_label,
        )
        self.transform_functions = [
            AddGaussianNoise(
                mean=gaussian_noise_mean,
                std=gaussian_noise_std,
                bernoulli_beta=train_bernoulli_beta,
            )
        ] + self.transform_functions
        self.transform = T.Compose(self.transform_functions)


class ContentAwareProcessor(Processor):
    return_keys = [
        "idx",
        "labels",
        "bboxes",
        "gold_bboxes",
        "content_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
        "discrete_content_bboxes",
    ]

    def __init__(
        self,
        index2label: dict,
        canvas_width: int,
        canvas_height: int,
        metadata: DataFrame,
        sort_by_pos: bool = False,
        shuffle_before_sort_by_label: bool = False,
        sort_by_pos_before_sort_by_label: bool = True,
        filter_threshold: int = 100,
        max_element_numbers: int = 10,
        original_width: float = 513.0,
        original_height: float = 750.0,
    ):
        super().__init__(
            index2label=index2label,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            sort_by_pos=sort_by_pos,
            shuffle_before_sort_by_label=shuffle_before_sort_by_label,
            sort_by_pos_before_sort_by_label=sort_by_pos_before_sort_by_label,
        )
        self.transform = T.Compose(self.transform_functions)
        self.metadata = metadata
        self.max_element_numbers = max_element_numbers
        self.original_width = original_width
        self.original_height = original_height
        self.saliency_map_to_bboxes = SaliencyMapToBBoxes(filter_threshold)
        self.possible_labels: list = []

    def _normalize_bboxes(self, bboxes):
        bboxes = bboxes.float()
        bboxes[:, 0::2] /= self.original_width
        bboxes[:, 1::2] /= self.original_height
        return bboxes

    def __call__(self, filename, idx, split):
        saliency_map = cv2.imread(filename)
        content_bboxes = self.saliency_map_to_bboxes(saliency_map)
        if len(content_bboxes) == 0:
            return None
        content_bboxes = self._normalize_bboxes(content_bboxes)

        if split == "train":
            _metadata = self.metadata[
                self.metadata["poster_path"] == f"train/{idx}.png"
            ][self.metadata["cls_elem"] > 0]
            labels = torch.tensor(list(map(int, _metadata["cls_elem"])))
            bboxes = torch.tensor(list(map(eval, _metadata["box_elem"])))
            if len(labels) == 0:
                return None
            bboxes[:, 2] -= bboxes[:, 0]
            bboxes[:, 3] -= bboxes[:, 1]
            bboxes = self._normalize_bboxes(bboxes)
            if len(labels) <= self.max_element_numbers:
                self.possible_labels.append(labels)

            data = {
                "idx": idx,
                "labels": labels,
                "bboxes": bboxes,
                "content_bboxes": content_bboxes,
            }
        else:
            if len(self.possible_labels) == 0:
                raise RuntimeError("Please process training data first")

            labels = random.choice(self.possible_labels)
            data = {
                "idx": idx,
                "labels": labels,
                "bboxes": torch.zeros((len(labels), 4)),  # dummy
                "content_bboxes": content_bboxes,
            }

        return super().__call__(data)


class TextToLayoutProcessor(Processor):
    return_keys = [
        "labels",
        "bboxes",
        "text",
        "embedding",
    ]

    def __init__(
        self,
        index2label: dict,
        canvas_width: int,
        canvas_height: int,
    ):
        self.index2label = index2label
        self.label2index = {v: k for k, v in self.index2label.items()}
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.text_encoder = CLIPTextEncoder()

    def _scale(self, original_width, elements_):
        elements = copy.deepcopy(elements_)
        ratio = self.canvas_width / original_width
        for i in range(len(elements)):
            elements[i]["position"][0] = int(ratio * elements[i]["position"][0])
            elements[i]["position"][1] = int(ratio * elements[i]["position"][1])
            elements[i]["position"][2] = int(ratio * elements[i]["position"][2])
            elements[i]["position"][3] = int(ratio * elements[i]["position"][3])
        return elements

    def __call__(self, data):
        text = clean_text(data["text"])
        embedding = self.text_encoder(clean_text(data["text"], remove_summary=True))
        original_width = data["canvas_width"]
        elements = data["elements"]
        elements = self._scale(original_width, elements)
        elements = sorted(elements, key=lambda x: (x["position"][1], x["position"][0]))

        labels = [self.label2index[element["type"]] for element in elements]
        labels = torch.tensor(labels)
        bboxes = [element["position"] for element in elements]
        bboxes = torch.tensor(bboxes)
        return {
            "text": text,
            "embedding": embedding,
            "labels": labels,
            "discrete_gold_bboxes": bboxes,
            "discrete_bboxes": bboxes,
        }


PROCESSOR_MAP = {
    "gent": GenTypeProcessor,
    "gents": GenTypeSizeProcessor,
    "genr": GenRelationProcessor,
    "completion": CompletionProcessor,
    "refinement": RefinementProcessor,
    "content": ContentAwareProcessor,
    "text": TextToLayoutProcessor,
}


def create_processor(dataset, task, *args, **kwargs):
    processor_cls = PROCESSOR_MAP[task]
    index2label = ID2LABEL[dataset]
    canvas_width, canvas_height = CANVAS_SIZE[dataset]
    processor = processor_cls(
        index2label=index2label,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        *args,
        **kwargs,
    )
    return processor
