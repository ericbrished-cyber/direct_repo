from __future__ import annotations

import os
from typing import Any, Dict, Optional
import base64

import httpx
from anthropic import Anthropic
from openai import OpenAI


def _infer_provider(model: str) -> str:
    lowered = model.lower()
    if lowered.startswith(("gpt", "o1")):
        return "openai"
    if lowered.startswith("claude"):
        return "anthropic"
    if lowered.startswith("gemini"):
        return "google"
    return "openai"


class LLMClient:
    """
    Thin wrapper around provider SDKs so the pipeline can stay provider-agnostic.
    Currently supports OpenAI Responses; raises helpful errors for other providers.
    """

    def __init__(
        self,
        model: str,
        provider: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.model = model
        self.provider = provider or _infer_provider(model)
        self.temperature = temperature

        if self.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY is not set")
            self._client = OpenAI(api_key=api_key)
        elif self.provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY is not set")
            self._client = Anthropic(api_key=api_key)
        elif self.provider == "google":
            raise NotImplementedError("Provider 'google' not implemented yet.")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _parse_openai_response(self, response: Any) -> str:
        """
        Extract the first text block from an OpenAI Responses API call.
        """
        output = getattr(response, "output", None)
        if not output:
            raise ValueError("OpenAI response contained no output blocks")
        for item in output:
            content = getattr(item, "content", None)
            if not content:
                continue
            for chunk in content:
                text = getattr(chunk, "text", None)
                if text is not None:
                    return text
        raise ValueError("OpenAI response did not contain text content")

    def generate(self, prompt: str, pdf_path: Optional[str] = None, **kwargs: Dict[str, Any]) -> str:
        """
        Execute a text-only prompt and return the raw string output.
        """
        if self.provider == "openai":
            attachments = []
            file_obj = None
            if pdf_path:
                file_obj = self._client.files.create(file=open(pdf_path, "rb"), purpose="user_data")
                attachments.append({"type": "input_file", "file_id": file_obj.id})

            messages = [{"role": "user", "content": [{"type": "input_text", "text": prompt}, *attachments]}]
            response = self._client.responses.create(
                model=self.model,
                input=messages,
                temperature=kwargs.get("temperature", self.temperature),
            )
            return self._parse_openai_response(response)

        if self.provider == "anthropic":
            content_blocks = [{"type": "text", "text": prompt}]
            if pdf_path:
                # Inline PDF as base64 to enable vision analysis (Claude messages API expects base64/url/content)
                with open(pdf_path, "rb") as f:
                    pdf_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
                content_blocks.insert(
                    0,
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    },
                )

            response = self._client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 4000),
                temperature=kwargs.get("temperature", self.temperature),
                messages=[{"role": "user", "content": content_blocks}],
            )
            for block in response.content or []:
                text = getattr(block, "text", None)
                if text:
                    return text
            raise ValueError("Anthropic response did not contain text content")

        raise NotImplementedError(f"Provider '{self.provider}' not implemented")
