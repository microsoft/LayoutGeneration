# coding=utf8

import re
from typing import Dict


class IRProcessor:

    TAG_RENAMES = [
        ("gp:navgroups", "gp:navgroup"),
        ("region:singleinfo", "region:info"),
        ("gattr:", "group_prop:"),
        ("attr:", "prop:"),
        ("gp:", "group:"),
        ("el:", "element:"),
    ]

    def __init__(self,
                 remove_value: bool = True,
                 replace_value: bool = False) -> None:
        self.remove_value = remove_value
        self.replace_value = replace_value

    def preprocess(self, lf: str, idx2value: Dict = None) -> str:
        # NOTE: fix error
        result = lf.replace("[", " [ ").replace("]", " ] ").strip().lower()
        result = re.sub(r"attr:(.*?)'", r"attr:\g<1> '", result)

        if self.remove_value:
            result = re.sub(r"\[\s*attr:value\s*'.*?'\s*\]", r"", result)
            result = re.sub(r"\[\s*gattr:names\s*'.*?'\s*\]", r"", result)

        for tag, new_tag in self.TAG_RENAMES:
            result = result.replace(tag, new_tag)

        result = result.replace(':', ' : ')
        result = re.sub("\s+", " ", result)

        if not self.remove_value and self.replace_value and idx2value is not None:
            value_map = {
                v.replace("`", "").replace("'", ""): k
                for k, v in idx2value.items()
            }
            result = self.replace_explicit_values(result, value_map)

        return result

    def replace_explicit_values(self, lf: str, value_map: Dict) -> str:
        # prop
        result = lf
        value_attrs = re.findall(r"(\[ prop:value '(.*?)' \])", lf)
        for attr, value in value_attrs:
            _value = value.strip()
            if _value in value_map:
                result = result.replace(
                    attr, f"[ prop:value '{value_map[_value]}' ]")
            else:
                # strip
                _value = ", ".join([_v.strip() for _v in _value.split(",")])
                result = result.replace(attr, f"[ prop:value '{_value}' ]")

        # names
        name_attrs = re.findall(r"(\[ group_prop:names '(.*?)' \])", lf)
        for attr, value in name_attrs:
            names = [n.strip() for n in value.split(",")]
            new_names = list()
            for name in names:
                if name in value_map:
                    new_names.append(value_map[name])
                else:
                    new_names.append(name)
            new_names = ", ".join(new_names)
            result = result.replace(attr, f"[ group_prop:names '{new_names}' ]")
        return result

    def postprocess(self,
                    lf: str,
                    remove_attrs: bool = False,
                    recover_labels: bool = False,
                    recover_values: bool = False,
                    value_map: Dict = None) -> str:
        result = lf.replace("[", " [ ").replace("]", " ] ").strip().lower()
        result = result.replace(" : ", ":")
        result = re.sub(r"prop:(.*?)'", r"prop:\g<1> '", result)
        if remove_attrs:
            result = re.sub(r"\[\s*prop:(value|size|position|repeat)\s*'.*?'\s*\]", r"", result)
        if recover_labels:
            for tag, new_tag in self.TAG_RENAMES[1:]:
                result = result.replace(new_tag, tag)
            for tag, new_tag in [("region:singleinfo", "region:SingleInfo"),
                                 ("region:multiinfo", "region:MultiInfo"),
                                 ("region:header", "region:Header"),
                                 ("region:footer", "region:Footer"),
                                 ("gp:infogroup", "gp:InfoGroup"),
                                 ("gp:navgroup", "gp:NavGroup"),
                                 ("gp:flatnavgroup", "gp:FlatNavGroup")]:
                result = result.replace(tag, new_tag)
            result = re.sub(r'\s*\[\s*', '[', result)
            result = re.sub(r'\s*\]\s*', ']', result)
            if recover_values and value_map is not None:
                for idx, value in value_map.items():
                    _value = value.replace("'", "")
                    result = result.replace(f"'{idx}'", f"'{_value}'")
                    result = result.replace(f" {idx},", f" {_value},")
                    result = result.replace(f"'{idx},", f"'{_value},")
                    result = result.replace(f" {idx}'", f" {_value}'")
                # replace &
                result = result.replace("&", " and ")
        result = re.sub("\s+", " ", result)
        return result
