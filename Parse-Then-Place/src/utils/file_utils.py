# coding=utf8

import os
import errno
import os.path as osp
import json
from typing import List, Dict


def read_jsonl(path: str) -> List[Dict]:
    data = list()
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            data.append(json.loads(line))
    return data


def write_jsonl(path: str, data) -> None:
    with open(path, 'w', encoding='utf8') as f:
        for item in data:
            f.write("{}\n".format(json.dumps(item)))


def read_json(path: str):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def write_json(path: str, data, indent: int=2) -> None:
    with open(path, 'w', encoding='utf8') as f:
        f.write(json.dumps(data, indent=2))


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e
