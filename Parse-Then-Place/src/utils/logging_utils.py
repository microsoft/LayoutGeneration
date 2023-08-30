# coding=utf8

import sys
import logging

from transformers.utils.logging import log_levels


def config_logger(log_level: str):
    _log_level = log_levels[log_level]
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=_log_level
    )
    return _log_level
