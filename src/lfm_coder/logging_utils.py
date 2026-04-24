import datetime
import json
import logging
import time
from pathlib import Path


class JSONFormatter(logging.Formatter):
    COMMON_ATTRS = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
        "message",
    }

    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "path": record.pathname,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        # Dynamically add "extra" attributes (items in record.__dict__ but not in COMMON_ATTRS)
        for key, value in record.__dict__.items():
            if key not in self.COMMON_ATTRS:
                log_entry[key] = value

        # Handle exceptions and stack traces
        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            log_entry["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(log_entry)

    def formatTime(self, record, datefmt=None):
        return time.strftime(
            datefmt or "%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)
        )


def _setup_logging():
    # Set up the root logger with handlers, so other loggers automatically inherit these settings
    logger = logging.getLogger()
    Path("logs").mkdir(parents=True, exist_ok=True)
    file_name = Path(
        "logs/log_"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        + ".jsonl"
    )
    file_handler = logging.FileHandler(file_name, mode="w", delay=True)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(JSONFormatter())

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    # Set the root logger to WARNING to suppress most third-party library logs
    logger.setLevel(logging.WARNING)

    # Explicitly set project loggers to DEBUG to catch all logs
    logging.getLogger("lfm_coder").setLevel(logging.DEBUG)
    logging.getLogger("__main__").setLevel(logging.DEBUG)

    return logger


logger = _setup_logging()


def get_logger(name):
    """Create a logger with the provided name. Logs all messages as JSON."""
    return logger.getChild(name)
