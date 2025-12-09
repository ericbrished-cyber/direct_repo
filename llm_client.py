from __future__ import annotations

import os
from typing import Any, Dict, Optional
import base64

import httpx
from anthropic import Anthropic
from openai import OpenAI
from google import genai


def _infer_provider(model: str) -> str:
    lowered = model.lower()
    if lowered.startswith("gpt"):
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
            # Use an explicit httpx client to avoid incompatible defaults between the SDK and httpx>=0.28.
            self._client = Anthropic(api_key=api_key, http_client=httpx.Client())
        elif self.provider == "google":
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError("GOOGLE_API_KEY is not set")
            self._client = genai.Client(api_key=api_key)
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

    def generate(
        self,
        prompt: str,
        pdf_path: Optional[str] = None,
        pdf_paths: Optional[list[str]] = None,
        num_fewshot_pdfs: int = 0,
        **kwargs: Dict[str, Any],
    ) -> str:
        """
        Execute a text-only prompt and return the raw string output.

        :param num_fewshot_pdfs: The number of PDFs at the start of pdf_paths that are 'few-shot' examples
                                 and should be cached (if supported by provider).
        """
        pdf_list = pdf_paths or ([] if pdf_path is None else [pdf_path])

        if self.provider == "openai":
            attachments = []
            file_objs = []
            for path in pdf_list:
                file_obj = self._client.files.create(file=open(path, "rb"), purpose="user_data")
                file_objs.append(file_obj)
                attachments.append({"type": "input_file", "file_id": file_obj.id})

            messages = [{"role": "user", "content": [{"type": "input_text", "text": prompt}, *attachments]}]
            max_tokens = kwargs.get("max_tokens")
            response = self._client.responses.create(
                model=self.model,
                input=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_output_tokens=max_tokens,
            )
            return self._parse_openai_response(response)

        if self.provider == "anthropic":
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
                # Only the last item in a contiguous cached sequence needs the marker,
                # but for simplicity/safety with PDF blocks, marking the break point is key.
                # If we have [FS1, FS2, Target], we cache FS2.
                # This caches FS1 and FS2.
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
                max_tokens=kwargs.get("max_tokens", 4000),
                temperature=kwargs.get("temperature", self.temperature),
                messages=[{"role": "user", "content": content_blocks}],
                extra_headers=extra_headers if extra_headers else None,
            )
            for block in response.content or []:
                text = getattr(block, "text", None)
                if text:
                    return text
            raise ValueError("Anthropic response did not contain text content")

        if self.provider == "google":
            file_refs = []
            for path in pdf_list:
                file_refs.append(self._client.files.upload(file=path))

            contents = []
            contents.extend(file_refs)
            contents.append(prompt)

            gen_config = kwargs.get("config")
            if gen_config is None:
                temperature = kwargs.get("temperature", self.temperature)
                max_output_tokens = kwargs.get("max_output_tokens")
                cfg_kwargs = {}
                # Only include keys when explicitly set to avoid overriding client defaults.
                if temperature is not None:
                    cfg_kwargs["temperature"] = temperature
                if max_output_tokens is not None:
                    cfg_kwargs["max_output_tokens"] = max_output_tokens
                gen_config = genai.types.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None

            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=gen_config,
            )

            text = getattr(response, "text", None)
            if text:
                return text

            for candidate in getattr(response, "candidates", []) or []:
                content = getattr(candidate, "content", None)
                if not content:
                    continue
                for part in getattr(content, "parts", []) or []:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        return part_text
                    if isinstance(part, dict):
                        part_text = part.get("text")
                        if part_text:
                            return part_text

            raise ValueError("Google response did not contain text content")

        raise NotImplementedError(f"Provider '{self.provider}' not implemented")
