"""Custom exceptions for the betting intelligence platform."""


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""


class StorageError(Exception):
    """Raised when storage layer operations fail."""


class DataValidationError(Exception):
    """Raised when data validation fails."""


class ApiError(Exception):
    """Raised when an external API call fails after all retries."""


class SchedulerError(Exception):
    """Raised when APScheduler job registration or execution fails."""


class ClvError(Exception):
    """Raised when CLV calculation or recording fails."""


class PickError(Exception):
    """Raised when pick engine evaluation or persistence fails."""


class ClaudeError(Exception):
    """Raised when the Claude Role C validator hits an unrecoverable error.

    Distinct from ApiError (network) — ClaudeError is raised when the validator
    returned a malformed response that strict tool_use should have prevented.
    Bug-class signal: investigate immediately if seen in production logs.
    """


class TelegramError(Exception):
    """Raised when the Telegram bot init/start/send pipeline fails."""
