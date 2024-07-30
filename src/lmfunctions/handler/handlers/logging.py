import logging
from typing import List, Literal, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.pretty import pretty_repr

from lmfunctions.handler.handler import Handler


class LoggingHandler(Handler):
    """
    Callback handler that log events to a logger.

    Attributes:
        name (Literal["logging"]): The name of the handler.
        logger_name (Optional[str]): The name of the logger to use.
        If not provided, a default logger will be used.
        log_level (int): The log level to use for logging events.
        log_time_format (str): The format string for the log timestamp.
        log_file (Optional[str]): The path to the log file.
        If provided, file logging will be used.
        file_logging_format (str): The format string for file logging.
        message (str): The message to include in the log event.
        log_keys (List[str]): The keys of the variables to include in the log event.

    """

    name: Literal["logging"] = "logging"
    logger_name: Optional[str] = "DEFAULT_LOGGER"
    log_level: int = logging.INFO
    log_time_format: str = "%Y-%m-%d %H:%M:%S"
    log_file: Optional[str] = None
    file_logging_format: str = (
        "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s"
    )
    message: str = "Logging event"
    log_keys: List[str] = []

    _logger: Optional[logging.Logger] = None

    @property
    def logger(self):
        if not self._logger:
            logger = logging.getLogger(self.logger_name)
            logger.setLevel(self.log_level)
            if self.log_file:
                # Create a logger for file logging
                handler = logging.FileHandler(self.log_file)
                handler.setFormatter(
                    logging.Formatter(
                        fmt=self.file_logging_format,
                        datefmt=self.log_time_format,
                    )
                )
            else:
                # Create a logger for stdout with rich handler
                handler = RichHandler(
                    console=Console(),
                    rich_tracebacks=True,
                    log_time_format=self.log_time_format,
                )
            logger.handlers = [handler]
            self._logger = logger
        return self._logger

    def __call__(self, **kwargs):
        filtered_vars = {key: kwargs[key] for key in self.log_keys if key in kwargs}
        pretty_vars = pretty_repr(filtered_vars)
        log_message = f"{self.message}\n{pretty_vars}"
        self.logger.log(self.log_level, log_message)
