import logging

PYLIZ_MEDIA_LOGGER_NAME = "PylizMedia"
logger = logging.getLogger(PYLIZ_MEDIA_LOGGER_NAME)
logger.addHandler(logging.NullHandler())