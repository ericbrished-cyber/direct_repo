import base64
import os
from typing import Any, Dict, Optional, List
import httpx
from anthropic import Anthropic
from .base import BaseLLMClient

class AnthropicClient(BaseLLMClient):
    def __init__(self, model: str, temperature: float = 0.0):
        super().__init__(model, temperature)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set")
        # Use an explicit httpx client to avoid incompatible defaults between the SDK and httpx>=0.28.
        self._client = Anthropic(api_key=api_key, http_client=httpx.Client())

    def generate(
        self,
        prompt: str,
        pdf_paths: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        num_fewshot_pdfs: int = 0,
        **kwargs: Dict[str, Any],
    ) -> str:
        pdf_list = pdf_paths or []
        content_blocks = []

        # Add PDF blocks
        for i, path in enumerate(pdf_list):
            with open(path, "rb") as f:
                pdf_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

            block = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_b64,
                },
            }

            # Apply caching to the last few-shot PDF.
            if i == num_fewshot_pdfs - 1 and num_fewshot_pdfs > 0:
                block["cache_control"] = {"type": "ephemeral"}

            content_blocks.append(block)

        content_blocks.append({"type": "text", "text": prompt})

        extra_headers = {}
        # Enable beta prompt caching headers if we are using caching
        if num_fewshot_pdfs > 0:
            extra_headers["anthropic-beta"] = "prompt-caching-2024-07-31"

        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens if max_tokens is not None else kwargs.get("max_tokens", 4000),
            temperature=kwargs.get("temperature", self.temperature),
            messages=[{"role": "user", "content": content_blocks}],
            extra_headers=extra_headers if extra_headers else None,
        )
        for block in response.content or []:
            text = getattr(block, "text", None)
            if text:
                return text
        raise ValueError("Anthropic response did not contain text content")
