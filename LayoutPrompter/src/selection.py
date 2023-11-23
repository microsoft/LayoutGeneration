import random

import cv2
import numpy as np

from utils import CANVAS_SIZE, labels_bboxes_similarity, labels_similarity


class ExemplarSelection:
    def __init__(
        self,
        train_data: list,
        candidate_size: int,
        num_prompt: int,
        shuffle: bool = True,
    ):
        self.train_data = train_data
        self.candidate_size = candidate_size
        self.num_prompt = num_prompt
        self.shuffle = shuffle
        if self.candidate_size > 0:
            random.shuffle(self.train_data)
            self.train_data = self.train_data[: self.candidate_size]

    def __call__(self, test_data: dict):
        pass

    def _is_filter(self, data):
        return (data["discrete_gold_bboxes"][:, 2:] == 0).sum().bool().item()

    def _retrieve_exemplars(self, scores: list):
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        exemplars = []
        for i in range(len(self.train_data)):
            if not self._is_filter(self.train_data[scores[i][0]]):
                exemplars.append(self.train_data[scores[i][0]])
                if len(exemplars) == self.num_prompt:
                    break
        if self.shuffle:
            random.shuffle(exemplars)
        return exemplars


class GenTypeExemplarSelection(ExemplarSelection):
    def __call__(self, test_data: dict):
        scores = []
        test_labels = test_data["labels"]
        for i in range(len(self.train_data)):
            train_labels = self.train_data[i]["labels"]
            score = labels_similarity(train_labels, test_labels)
            scores.append([i, score])
        return self._retrieve_exemplars(scores)


class GenTypeSizeExemplarSelection(ExemplarSelection):
    labels_weight = 0.5
    bboxes_weight = 0.5

    def __call__(self, test_data: dict):
        scores = []
        test_labels = test_data["labels"]
        test_bboxes = test_data["bboxes"][:, 2:]
        for i in range(len(self.train_data)):
            train_labels = self.train_data[i]["labels"]
            train_bboxes = self.train_data[i]["bboxes"][:, 2:]
            score = labels_bboxes_similarity(
                train_labels,
                train_bboxes,
                test_labels,
                test_bboxes,
                self.labels_weight,
                self.bboxes_weight,
            )
            scores.append([i, score])
        return self._retrieve_exemplars(scores)


class GenRelationExemplarSelection(ExemplarSelection):
    def __call__(self, test_data: dict):
        scores = []
        test_labels = test_data["labels"]
        for i in range(len(self.train_data)):
            train_labels = self.train_data[i]["labels"]
            score = labels_similarity(train_labels, test_labels)
            scores.append([i, score])
        return self._retrieve_exemplars(scores)


class CompletionExemplarSelection(ExemplarSelection):
    labels_weight = 0.0
    bboxes_weight = 1.0

    def __call__(self, test_data: dict):
        scores = []
        test_labels = test_data["labels"][:1]
        test_bboxes = test_data["bboxes"][:1, :]
        for i in range(len(self.train_data)):
            train_labels = self.train_data[i]["labels"][:1]
            train_bboxes = self.train_data[i]["bboxes"][:1, :]
            score = labels_bboxes_similarity(
                train_labels,
                train_bboxes,
                test_labels,
                test_bboxes,
                self.labels_weight,
                self.bboxes_weight,
            )
            scores.append([i, score])
        return self._retrieve_exemplars(scores)


class RefinementExemplarSelection(ExemplarSelection):
    labels_weight = 0.5
    bboxes_weight = 0.5

    def __call__(self, test_data: dict):
        scores = []
        test_labels = test_data["labels"]
        test_bboxes = test_data["bboxes"]
        for i in range(len(self.train_data)):
            train_labels = self.train_data[i]["labels"]
            train_bboxes = self.train_data[i]["bboxes"]
            score = labels_bboxes_similarity(
                train_labels,
                train_bboxes,
                test_labels,
                test_bboxes,
                self.labels_weight,
                self.bboxes_weight,
            )
            scores.append([i, score])
        return self._retrieve_exemplars(scores)


class ContentAwareExemplarSelection(ExemplarSelection):
    canvas_width, canvas_height = CANVAS_SIZE["posterlayout"]

    def _to_binary_image(self, content_bboxes):
        binary_image = np.zeros((self.canvas_height, self.canvas_width), dtype=np.uint8)
        content_bboxes = content_bboxes.tolist()
        for content_bbox in content_bboxes:
            l, t, w, h = content_bbox
            cv2.rectangle(binary_image, (l, t), (l + w, t + h), 255, thickness=-1)
        return binary_image

    def __call__(self, test_data: dict):
        scores = []
        test_content_bboxes = test_data["discrete_content_bboxes"]
        test_binary = self._to_binary_image(test_content_bboxes)
        for i in range(len(self.train_data)):
            train_content_bboxes = self.train_data[i]["discrete_content_bboxes"]
            train_binary = self._to_binary_image(train_content_bboxes)
            intersection = cv2.bitwise_and(train_binary, test_binary)
            union = cv2.bitwise_or(train_binary, test_binary)
            iou = (np.sum(intersection) + 1) / (np.sum(union) + 1)
            scores.append([i, iou])
        return self._retrieve_exemplars(scores)


class TextToLayoutExemplarSelection(ExemplarSelection):
    def __call__(self, test_data: dict):
        scores = []
        test_embedding = test_data["embedding"]
        for i in range(len(self.train_data)):
            train_embedding = self.train_data[i]["embedding"]
            score = (train_embedding @ test_embedding.T).item()
            scores.append([i, score])
        return self._retrieve_exemplars(scores)


SELECTOR_MAP = {
    "gent": GenTypeExemplarSelection,
    "gents": GenTypeSizeExemplarSelection,
    "genr": GenRelationExemplarSelection,
    "completion": CompletionExemplarSelection,
    "refinement": RefinementExemplarSelection,
    "content": ContentAwareExemplarSelection,
    "text": TextToLayoutExemplarSelection,
}


def create_selector(task, train_data, candidate_size, num_prompt, *args, **kwargs):
    selector_cls = SELECTOR_MAP[task]
    selector = selector_cls(
        train_data=train_data,
        candidate_size=candidate_size,
        num_prompt=num_prompt,
        *args,
        **kwargs,
    )
    return selector
