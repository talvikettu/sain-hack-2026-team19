"""RAG orchestration, independent of the eventual frontend/API framework."""

from .agent import RAGAgent, ValidationError, validate_answer
from .client import (
    BudgetError, ChatClient, ChatError, ChatTerminalError, ChatTransientError,
    ProxyProtocolError,
)

__all__ = [
    "RAGAgent", "ValidationError", "validate_answer", "ChatClient",
    "ChatError", "ChatTransientError", "ChatTerminalError", "BudgetError",
    "ProxyProtocolError",
]
