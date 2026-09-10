# encoding:utf8
"""Runtime logging configuration for the Zerith pose server."""

import logging
from pathlib import Path


LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d [%(levelname)s] "
    "%(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_runtime_logging(log_file):
    """Add a millisecond-resolution file log while retaining console logs.

    The file is opened in append mode so consecutive server runs remain
    separated by their ``SERVER_START`` events. Repeated calls for the same
    path do not install duplicate handlers, which also makes this safe in
    tests and embedded launchers.
    """
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # ``Utils`` installs the console handler during import. Upgrade it to the
    # same millisecond-resolution format used by the runtime file.
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

    if not log_file:
        return None

    path = Path(log_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    for handler in root_logger.handlers:
        if not isinstance(handler, logging.FileHandler):
            continue
        if Path(handler.baseFilename).resolve() == path:
            return path

    file_handler = logging.FileHandler(
        str(path), mode="a", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    return path
