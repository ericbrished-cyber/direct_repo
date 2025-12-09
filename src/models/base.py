from abc import ABC, abstractmethod
from typing import Tuple, Dict
from src.prompts.builder import PromptPayload

class ModelAdapter(ABC):
    """
    Abstract Base Class for Model Adapters.
    """

    @abstractmethod
    def generate(self, payload: PromptPayload, temperature: float = 0.0) -> Tuple[str, Dict[str, int]]:
        """
        Generates text based on the payload.
        Args:
            payload: The prompt payload object.
            temperature: The temperature for generation (0.0 to 1.0).
        Returns:
            raw_text (str): The model's response text.
            token_usage_dict (dict): Dictionary containing token usage stats (e.g., {'input': 100, 'output': 50}).
        """
        pass
