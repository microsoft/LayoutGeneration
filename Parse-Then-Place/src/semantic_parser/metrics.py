# coding=utf8

from abc import ABC, abstractmethod
from typing import Dict


class Metric(ABC):

    def __init__(self) -> None:
        self._is_positive = True

    @abstractmethod
    def aggregate(self, metrics: Dict):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @property
    def is_positive(self):
        return self._is_positive


class LossMetric(Metric):

    METRIC_NAME = 'loss'

    def __init__(self) -> None:
        super().__init__()
        self.num_batch = 0
        self.total_loss = 0.0
        self._is_positive = False

    def aggregate(self, metrics: Dict) -> None:
        self.num_batch += 1
        self.total_loss += metrics['loss'].item()

    def compute(self) -> Dict:
        if self.num_batch == 0:
            return {self.METRIC_NAME: 0.0}
        return {self.METRIC_NAME: self.total_loss / self.num_batch}

    def reset(self) -> None:
        self.num_batch = 0
        self.total_loss = 0.0


class AccMetric(Metric):

    METRIC_NAME = 'accuracy'

    def __init__(self) -> None:
        super().__init__()
        self.num_correct = 0
        self.num_examples = 0.0

    def aggregate(self, metrics: Dict) -> None:
        self.num_correct += metrics.get('num_correct', 0)
        self.num_examples += metrics.get('num_examples', 0)

    def compute(self) -> Dict:
        if self.num_examples == 0:
            return {self.METRIC_NAME: 0.0}
        return {self.METRIC_NAME: self.num_correct / self.num_examples}

    def reset(self) -> None:
        self.num_correct = 0
        self.num_examples = 0.0


class ElementAccMetric(Metric):

    METRIC_NAME = 'element_accuracy'

    def __init__(self) -> None:
        super().__init__()
        self.num_correct = 0
        self.num_examples = 0.0

    def aggregate(self, metrics: Dict) -> None:
        self.num_correct += metrics.get('num_element_correct', 0)
        self.num_examples += metrics.get('num_examples', 0)

    def compute(self) -> Dict:
        if self.num_examples == 0:
            return {self.METRIC_NAME: 0.0}
        return {self.METRIC_NAME: self.num_correct / self.num_examples}

    def reset(self) -> None:
        self.num_correct = 0
        self.num_examples = 0.0


class SetAccMetric(Metric):

    METRIC_NAME = 'set_accuracy'

    def __init__(self) -> None:
        super().__init__()
        self.num_correct = 0
        self.num_examples = 0.0

    def aggregate(self, metrics: Dict) -> None:
        self.num_correct += metrics.get('num_set_correct', 0)
        self.num_examples += metrics.get('num_examples', 0)

    def compute(self) -> Dict:
        if self.num_examples == 0:
            return {self.METRIC_NAME: 0.0}
        return {self.METRIC_NAME: self.num_correct / self.num_examples}

    def reset(self) -> None:
        self.num_correct = 0
        self.num_examples = 0.0

