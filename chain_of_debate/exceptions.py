"""
Custom exceptions for Chain of Debate.

This module defines custom exception classes used throughout the
Chain of Debate system to provide more specific error handling.
"""


class ChainOfDebateError(Exception):
    """Base exception class for Chain of Debate errors."""
    pass


class ConfigurationError(ChainOfDebateError):
    """Raised when there's an error in configuration files or parameters."""
    pass


class APIError(ChainOfDebateError):
    """Raised when there's an error with OpenAI API calls."""
    pass


class AgentError(ChainOfDebateError):
    """Raised when there's an error with agent responses or behavior."""
    pass


class ConsensusError(ChainOfDebateError):
    """Raised when agents fail to reach consensus after maximum rounds."""
    pass


class ModelError(ChainOfDebateError):
    """Raised when there's an error with AI model configuration or responses."""
    pass


class ValidationError(ChainOfDebateError):
    """Raised when validation of inputs or configurations fails."""
    pass


class FileError(ChainOfDebateError):
    """Raised when there's an error reading or writing files."""
    pass


class TokenLimitError(ChainOfDebateError):
    """Raised when token limits are exceeded."""
    pass


class TimeoutError(ChainOfDebateError):
    """Raised when operations timeout."""
    pass
