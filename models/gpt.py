import os
from typing import Any, Dict, Optional, List
from openai import OpenAI
from .base import BaseLLMClient

class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str, temperature: float = 0.0):
        super().__init__(model, temperature)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key)

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
        pdf_paths: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        num_fewshot_pdfs: int = 0,
        **kwargs: Dict[str, Any],
    ) -> str:
        pdf_list = pdf_paths or []
        attachments = []
        file_objs = []
        for path in pdf_list:
            file_obj = self._client.files.create(file=open(path, "rb"), purpose="user_data")
            file_objs.append(file_obj)
            attachments.append({"type": "input_file", "file_id": file_obj.id})

        messages = [{"role": "user", "content": [{"type": "input_text", "text": prompt}, *attachments]}]

        response = self._client.responses.create(
            model=self.model,
            input=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_output_tokens=max_tokens,
        )
        return self._parse_openai_response(response)
