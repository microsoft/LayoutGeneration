# coding=utf8

import re
from typing import Dict, Tuple


class TextProcessor:

    VALUE_PLACEHOLDER = 'value_'

    def __init__(self, replace_value: bool = False) -> None:
        self.replace_value = replace_value

    def preprocess(self, text: str) -> Tuple[str, Dict]:
        result = text.replace("#", "").strip().lower()
        result = result.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")
        result = re.sub("\s+", " ", result)
        value_map = None
        if self.replace_value:
            result, value_map = self.extract_explicit_values(result)
        return result, value_map

    def extract_explicit_values(self, text: str) -> Tuple:
        result = re.sub(r"(\w)'s\s+", r'\g<1>`s ', text)
        value_map = dict()
        values = re.findall(r'".*?"', result)
        values_1 = re.findall(r"'.*?'", result)
        if len(values_1) == 1 and any([punct in values_1[0] for punct in [",", "."]]):
            values_1 = list()
        values.extend(values_1)
        for vidx, v in enumerate(values):
            placeholder = f'{self.VALUE_PLACEHOLDER}{vidx}'
            value_map[placeholder] = v.strip('"').strip()
            result = result.replace(v, f'"{placeholder}"', 1) # replace one occurrence per time
        result = re.sub(r"(\w)`s\s+", r"\g<1>'s ", result)
        result = re.sub("\s+", " ", result)
        return result, value_map
