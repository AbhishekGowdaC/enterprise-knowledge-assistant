class DatabaseUnavailableError(Exception):
    """Raised when PostgreSQL is unavailable."""
    pass


class VectorDatabaseUnavailableError(Exception):
    """Raised when Qdrant is unavailable."""
    pass


class LLMServiceUnavailableError(Exception):
    """Raised when the LLM provider is unavailable."""
    pass

class LLMConfigurationError(Exception):
    """Raised when the LLM provider configuration is invalid."""
    pass