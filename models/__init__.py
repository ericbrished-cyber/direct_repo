from typing import Optional
from .base import BaseLLMClient
from .claude import AnthropicClient
from .gpt import OpenAIClient
from .gemini import GeminiClient

def get_client(model: str, temperature: float = 0.0) -> BaseLLMClient:
    lowered = model.lower()
    if lowered.startswith("claude"):
        return AnthropicClient(model, temperature)
    if lowered.startswith("gpt"):
        return OpenAIClient(model, temperature)
    if lowered.startswith("gemini"):
        return GeminiClient(model, temperature)
    # Default fallback
    return OpenAIClient(model, temperature)
