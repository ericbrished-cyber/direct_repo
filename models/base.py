from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.
    """

    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    @abstractmethod
    def generate(
        self,
        prompt: str,
        pdf_paths: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        num_fewshot_pdfs: int = 0,
        **kwargs: Dict[str, Any],
    ) -> str:
        """
        Execute a prompt and return the text output.
        """
        pass
