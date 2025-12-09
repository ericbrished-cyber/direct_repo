import os
from typing import Any, Dict, Optional, List
from google import genai
from .base import BaseLLMClient

class GeminiClient(BaseLLMClient):
    def __init__(self, model: str, temperature: float = 0.0):
        super().__init__(model, temperature)
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set")
        self._client = genai.Client(api_key=api_key)

    def generate(
        self,
        prompt: str,
        pdf_paths: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        num_fewshot_pdfs: int = 0,
        **kwargs: Dict[str, Any],
    ) -> str:
        pdf_list = pdf_paths or []
        file_refs = []
        for path in pdf_list:
            file_refs.append(self._client.files.upload(file=path))

        contents = []
        contents.extend(file_refs)
        contents.append(prompt)

        gen_config = kwargs.get("config")
        if gen_config is None:
            temperature = kwargs.get("temperature", self.temperature)
            max_output_tokens = max_tokens if max_tokens is not None else kwargs.get("max_output_tokens")
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
