# coding: utf-8

import logging


MESSAGE_FORMAT = (
    "%(levelname)s - %(name)s - %(asctime)s - %(threadName)s (%(thread)d):\n%(message)s"
)
DATE_FORMAT = "%d/%m/%Y %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=MESSAGE_FORMAT)
logger = logging.getLogger()
