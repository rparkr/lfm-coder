import json
import logging


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        return json.dumps(log_entry)


# Set up the root logger with handlers, so other loggers automatically inherit these settings
logger = logging.getLogger()
if not logger.handlers:
    file_handler = logging.FileHandler("logs.jsonl", mode="w", delay=True)
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


def get_logger(name):
    """Create a logger with the provided name. Logs all messages as JSON."""
    return logger.getChild(name)
