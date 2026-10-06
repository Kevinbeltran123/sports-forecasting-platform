"""Structured logging setup using structlog."""

import sys

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog for JSON in production, pretty in terminal."""
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    if sys.stderr.isatty():
        processors = shared_processors + [structlog.dev.ConsoleRenderer()]
    else:
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
