from transforms import RelationTypes
from utils import CANVAS_SIZE, ID2LABEL, LAYOUT_DOMAIN

PREAMBLE = (
    "Please generate a layout based on the given information. "
    "You need to ensure that the generated layout looks realistic, with elements well aligned and avoiding unnecessary overlap.\n"
    "Task Description: {}\n"
    "Layout Domain: {} layout\n"
    "Canvas Size: canvas width is {}px, canvas height is {}px"
)

HTML_PREFIX = """<html>
<body>
<div class="canvas" style="left: 0px; top: 0px; width: {}px; height: {}px"></div>
"""

HTML_SUFFIX = """</body>
</html>"""

HTML_TEMPLATE = """<div class="{}" style="left: {}px; top: {}px; width: {}px; height: {}px"></div>
"""

HTML_TEMPLATE_WITH_INDEX = """<div class="{}" style="index: {}; left: {}px; top: {}px; width: {}px; height: {}px"></div>
"""


class Serializer:
    def __init__(
        self,
        input_format: str,
        output_format: str,
        index2label: dict,
        canvas_width: int,
        canvas_height: int,
        add_index_token: bool = True,
        add_sep_token: bool = True,
        sep_token: str = "|",
        add_unk_token: bool = False,
        unk_token: str = "<unk>",
    ):
        self.input_format = input_format
        self.output_format = output_format
        self.index2label = index2label
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.add_index_token = add_index_token
        self.add_sep_token = add_sep_token
        self.sep_token = sep_token
        self.add_unk_token = add_unk_token
        self.unk_token = unk_token

    def build_input(self, data):
        if self.input_format == "seq":
            return self._build_seq_input(data)
        elif self.input_format == "html":
            return self._build_html_input(data)
        else:
            raise ValueError(f"Unsupported input format: {self.input_format}")

    def _build_seq_input(self, data):
        raise NotImplementedError

    def _build_html_input(self, data):
        raise NotImplementedError

    def build_output(self, data, label_key="labels", bbox_key="discrete_gold_bboxes"):
        if self.output_format == "seq":
            return self._build_seq_output(data, label_key, bbox_key)
        elif self.output_format == "html":
            return self._build_html_output(data, label_key, bbox_key)

    def _build_seq_output(self, data, label_key, bbox_key):
        labels = data[label_key]
        bboxes = data[bbox_key]
        tokens = []

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            bbox = bboxes[idx].tolist()
            tokens.append(label)
            if self.add_index_token:
                tokens.append(str(idx))
            tokens.extend(map(str, bbox))
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(self.sep_token)
        return " ".join(tokens)

    def _build_html_output(self, data, label_key, bbox_key):
        labels = data[label_key]
        bboxes = data[bbox_key]
        htmls = [HTML_PREFIX.format(self.canvas_width, self.canvas_height)]
        _TEMPLATE = HTML_TEMPLATE_WITH_INDEX if self.add_index_token else HTML_TEMPLATE

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            bbox = bboxes[idx].tolist()
            element = [label]
            if self.add_index_token:
                element.append(str(idx))
            element.extend(map(str, bbox))
            htmls.append(_TEMPLATE.format(*element))
        htmls.append(HTML_SUFFIX)
        return "".join(htmls)


class GenTypeSerializer(Serializer):
    task_type = "generation conditioned on given element types"
    constraint_type = ["Element Type Constraint: "]
    HTML_TEMPLATE_WITHOUT_ANK = '<div class="{}"></div>\n'
    HTML_TEMPLATE_WITHOUT_ANK_WITH_INDEX = '<div class="{}" style="index: {}"></div>\n'

    def _build_seq_input(self, data):
        labels = data["labels"]
        tokens = []

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            tokens.append(label)
            if self.add_index_token:
                tokens.append(str(idx))
            if self.add_unk_token:
                tokens += [self.unk_token] * 4
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(self.sep_token)
        return " ".join(tokens)

    def _build_html_input(self, data):
        labels = data["labels"]
        htmls = [HTML_PREFIX.format(self.canvas_width, self.canvas_height)]
        if self.add_index_token and self.add_unk_token:
            _TEMPLATE = HTML_TEMPLATE_WITH_INDEX
        elif self.add_index_token and not self.add_unk_token:
            _TEMPLATE = self.HTML_TEMPLATE_WITHOUT_ANK_WITH_INDEX
        elif not self.add_index_token and self.add_unk_token:
            _TEMPLATE = HTML_TEMPLATE
        else:
            _TEMPLATE = self.HTML_TEMPLATE_WITHOUT_ANK

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            element = [label]
            if self.add_index_token:
                element.append(str(idx))
            if self.add_unk_token:
                element += [self.unk_token] * 4
            htmls.append(_TEMPLATE.format(*element))
        htmls.append(HTML_SUFFIX)
        return "".join(htmls)

    def build_input(self, data):
        return self.constraint_type[0] + super().build_input(data)


class GenTypeSizeSerializer(Serializer):
    task_type = "generation conditioned on given element types and sizes"
    constraint_type = ["Element Type and Size Constraint: "]
    HTML_TEMPLATE_WITHOUT_ANK = (
        '<div class="{}" style="width: {}px; height: {}px"></div>\n'
    )
    HTML_TEMPLATE_WITHOUT_ANK_WITH_INDEX = (
        '<div class="{}" style="index: {}; width: {}px; height: {}px"></div>\n'
    )

    def _build_seq_input(self, data):
        labels = data["labels"]
        bboxes = data["discrete_gold_bboxes"]
        tokens = []

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            bbox = bboxes[idx].tolist()
            tokens.append(label)
            if self.add_index_token:
                tokens.append(str(idx))
            if self.add_unk_token:
                tokens += [self.unk_token] * 2
            tokens.extend(map(str, bbox[2:]))
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(self.sep_token)
        return " ".join(tokens)

    def _build_html_input(self, data):
        labels = data["labels"]
        bboxes = data["discrete_gold_bboxes"]
        htmls = [HTML_PREFIX.format(self.canvas_width, self.canvas_height)]
        if self.add_index_token and self.add_unk_token:
            _TEMPLATE = HTML_TEMPLATE_WITH_INDEX
        elif self.add_index_token and not self.add_unk_token:
            _TEMPLATE = self.HTML_TEMPLATE_WITHOUT_ANK_WITH_INDEX
        elif not self.add_index_token and self.add_unk_token:
            _TEMPLATE = HTML_TEMPLATE
        else:
            _TEMPLATE = self.HTML_TEMPLATE_WITHOUT_ANK

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            bbox = bboxes[idx].tolist()
            element = [label]
            if self.add_index_token:
                element.append(str(idx))
            if self.add_unk_token:
                element += [self.unk_token] * 2
            element.extend(map(str, bbox[2:]))
            htmls.append(_TEMPLATE.format(*element))
        htmls.append(HTML_SUFFIX)
        return "".join(htmls)

    def build_input(self, data):
        return self.constraint_type[0] + super().build_input(data)


class GenRelationSerializer(Serializer):
    task_type = (
        "generation conditioned on given element relationships\n"
        "'A left B' means that the center coordinate of A is to the left of the center coordinate of B. "
        "'A right B' means that the center coordinate of A is to the right of the center coordinate of B. "
        "'A top B' means that the center coordinate of A is above the center coordinate of B. "
        "'A bottom B' means that the center coordinate of A is below the center coordinate of B. "
        "'A center B' means that the center coordinate of A and the center coordinate of B are very close. "
        "'A smaller B' means that the area of A is smaller than the ares of B. "
        "'A larger B' means that the area of A is larger than the ares of B. "
        "'A equal B' means that the area of A and the ares of B are very close. "
        "Here, center coordinate = (left + width / 2, top + height / 2), "
        "area = width * height"
    )
    constraint_type = ["Element Type Constraint: ", "Element Relationship Constraint: "]
    HTML_TEMPLATE_WITHOUT_ANK = '<div class="{}"></div>\n'
    HTML_TEMPLATE_WITHOUT_ANK_WITH_INDEX = '<div class="{}" style="index: {}"></div>\n'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index2type = RelationTypes.index2type()

    def _build_seq_input(self, data):
        labels = data["labels"]
        relations = data["relations"]
        tokens = []

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            tokens.append(label)
            if self.add_index_token:
                tokens.append(str(idx))
            if self.add_unk_token:
                tokens += [self.unk_token] * 4
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(self.sep_token)
        type_cons = " ".join(tokens)
        if len(relations) == 0:
            return self.constraint_type[0] + type_cons
        tokens = []
        for idx in range(len(relations)):
            label_i = relations[idx][2]
            index_i = relations[idx][3]
            if label_i != 0:
                tokens.append("{} {}".format(self.index2label[int(label_i)], index_i))
            else:
                tokens.append("canvas")
            tokens.append(self.index2type[int(relations[idx][4])])
            label_j = relations[idx][0]
            index_j = relations[idx][1]
            if label_j != 0:
                tokens.append("{} {}".format(self.index2label[int(label_j)], index_j))
            else:
                tokens.append("canvas")
            if self.add_sep_token and idx < len(relations) - 1:
                tokens.append(self.sep_token)
        relation_cons = " ".join(tokens)
        return (
            self.constraint_type[0]
            + type_cons
            + "\n"
            + self.constraint_type[1]
            + relation_cons
        )

    def _build_html_input(self, data):
        labels = data["labels"]
        relations = data["relations"]
        htmls = [HTML_PREFIX.format(self.canvas_width, self.canvas_height)]
        if self.add_index_token and self.add_unk_token:
            _TEMPLATE = HTML_TEMPLATE_WITH_INDEX
        elif self.add_index_token and not self.add_unk_token:
            _TEMPLATE = self.HTML_TEMPLATE_WITHOUT_ANK_WITH_INDEX
        elif not self.add_index_token and self.add_unk_token:
            _TEMPLATE = HTML_TEMPLATE
        else:
            _TEMPLATE = self.HTML_TEMPLATE_WITHOUT_ANK

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            element = [label]
            if self.add_index_token:
                element.append(str(idx))
            if self.add_unk_token:
                element += [self.unk_token] * 4
            htmls.append(_TEMPLATE.format(*element))
        htmls.append(HTML_SUFFIX)
        type_cons = "".join(htmls)
        if len(relations) == 0:
            return self.constraint_type[0] + type_cons
        tokens = []
        for idx in range(len(relations)):
            label_i = relations[idx][2]
            index_i = relations[idx][3]
            if label_i != 0:
                tokens.append("{} {}".format(self.index2label[int(label_i)], index_i))
            else:
                tokens.append("canvas")
            tokens.append(self.index2type[int(relations[idx][4])])
            label_j = relations[idx][0]
            index_j = relations[idx][1]
            if label_j != 0:
                tokens.append("{} {}".format(self.index2label[int(label_j)], index_j))
            else:
                tokens.append("canvas")
            if self.add_sep_token and idx < len(relations) - 1:
                tokens.append(self.sep_token)
        relation_cons = " ".join(tokens)
        return (
            self.constraint_type[0]
            + type_cons
            + "\n"
            + self.constraint_type[1]
            + relation_cons
        )


class CompletionSerializer(Serializer):
    task_type = "layout completion"
    constraint_type = ["Partial Layout: "]

    def _build_seq_input(self, data):
        data["partial_labels"] = data["labels"][:1]
        data["partial_bboxes"] = data["discrete_bboxes"][:1, :]
        return self._build_seq_output(data, "partial_labels", "partial_bboxes")

    def _build_html_input(self, data):
        data["partial_labels"] = data["labels"][:1]
        data["partial_bboxes"] = data["discrete_bboxes"][:1, :]
        return self._build_html_output(data, "partial_labels", "partial_bboxes")

    def build_input(self, data):
        return self.constraint_type[0] + super().build_input(data)


class RefinementSerializer(Serializer):
    task_type = "layout refinement"
    constraint_type = ["Noise Layout: "]

    def _build_seq_input(self, data):
        return self._build_seq_output(data, "labels", "discrete_bboxes")

    def _build_html_input(self, data):
        return self._build_html_output(data, "labels", "discrete_bboxes")

    def build_input(self, data):
        return self.constraint_type[0] + super().build_input(data)


class ContentAwareSerializer(Serializer):
    task_type = (
        "content-aware layout generation\n"
        "Please place the following elements to avoid salient content, and underlay must be the background of text or logo."
    )
    constraint_type = ["Content Constraint: ", "Element Type Constraint: "]
    CONTENT_TEMPLATE = "left {}px, top {}px, width {}px, height {}px"

    def _build_seq_input(self, data):
        labels = data["labels"]
        content_bboxes = data["discrete_content_bboxes"]

        tokens = []
        for idx in range(len(content_bboxes)):
            content_bbox = content_bboxes[idx].tolist()
            tokens.append(self.CONTENT_TEMPLATE.format(*content_bbox))
            if self.add_index_token and idx < len(content_bboxes) - 1:
                tokens.append(self.sep_token)
        content_cons = " ".join(tokens)

        tokens = []
        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            tokens.append(label)
            if self.add_index_token:
                tokens.append(str(idx))
            if self.add_unk_token:
                tokens += [self.unk_token] * 4
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(self.sep_token)
        type_cons = " ".join(tokens)
        return (
            self.constraint_type[0]
            + content_cons
            + "\n"
            + self.constraint_type[1]
            + type_cons
        )


class TextToLayoutSerializer(Serializer):
    task_type = (
        "text-to-layout\n"
        "There are ten optional element types, including: image, icon, logo, background, title, description, text, link, input, button. "
        "Please do not exceed the boundaries of the canvas. "
        "Besides, do not generate elements at the edge of the canvas, that is, reduce top: 0px and left: 0px predictions as much as possible."
    )
    constraint_type = ["Text: "]

    def _build_seq_input(self, data):
        return data["text"]

    def build_input(self, data):
        return self.constraint_type[0] + super().build_input(data)


SERIALIZER_MAP = {
    "gent": GenTypeSerializer,
    "gents": GenTypeSizeSerializer,
    "genr": GenRelationSerializer,
    "completion": CompletionSerializer,
    "refinement": RefinementSerializer,
    "content": ContentAwareSerializer,
    "text": TextToLayoutSerializer,
}


def create_serializer(
    dataset,
    task,
    input_format,
    output_format,
    add_index_token,
    add_sep_token,
    add_unk_token,
):
    serializer_cls = SERIALIZER_MAP[task]
    index2label = ID2LABEL[dataset]
    canvas_width, canvas_height = CANVAS_SIZE[dataset]
    serializer = serializer_cls(
        input_format=input_format,
        output_format=output_format,
        index2label=index2label,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        add_index_token=add_index_token,
        add_sep_token=add_sep_token,
        add_unk_token=add_unk_token,
    )
    return serializer


def build_prompt(
    serializer,
    exemplars,
    test_data,
    dataset,
    max_length=8000,
    separator_in_samples="\n",
    separator_between_samples="\n\n",
):
    prompt = [
        PREAMBLE.format(
            serializer.task_type, LAYOUT_DOMAIN[dataset], *CANVAS_SIZE[dataset]
        )
    ]
    for i in range(len(exemplars)):
        _prompt = (
            serializer.build_input(exemplars[i])
            + separator_in_samples
            + serializer.build_output(exemplars[i])
        )
        if len(separator_between_samples.join(prompt) + _prompt) <= max_length:
            prompt.append(_prompt)
        else:
            break
    prompt.append(serializer.build_input(test_data) + separator_in_samples)
    return separator_between_samples.join(prompt)


if __name__ == "__main__":
    import torch

    from utils import ID2LABEL

    ls = RefinementSerializer(
        input_format="seq",
        output_format="html",
        index2label=ID2LABEL["publaynet"],
        canvas_width=120,
        canvas_height=160,
        add_sep_token=True,
        add_unk_token=False,
        add_index_token=True,
    )
    labels = torch.tensor([4, 4, 1, 1, 1, 1])
    bboxes = torch.tensor(
        [
            [29, 14, 59, 2],
            [10, 18, 99, 57],
            [10, 79, 99, 4],
            [10, 85, 99, 7],
            [10, 99, 47, 50],
            [61, 99, 47, 50],
        ]
    )

    rearranged_labels = torch.tensor([1, 4, 1, 4, 1, 1])
    relations = torch.tensor([[4, 1, 0, 1, 4], [1, 2, 1, 3, 2]])
    data = {
        "labels": labels,
        "discrete_bboxes": bboxes,
        "discrete_gold_bboxes": bboxes,
        "relations": relations,
        "rearranged_labels": rearranged_labels,
    }
    print("--------")
    print(ls.build_input(data))
    print("--------")
    print(ls.build_output(data))
