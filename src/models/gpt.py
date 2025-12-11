from typing import Tuple, Dict
from src.models.base import ModelAdapter
from src.prompts.builder import PromptPayload
import os
import base64
from openai import OpenAI

class GPTModel(ModelAdapter):
    """
    OpenAI GPT-5.1 implementation with native PDF support.
    """
    def __init__(self, model_version: str = "gpt-5.1"):
        self.model_version = model_version
        self.api_key = os.getenv("OPENAI_API_KEY")

    def _encode_pdf_to_base64(self, pdf_path: str) -> str:
        """Encodes PDF to base64."""
        with open(pdf_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate(self, payload: PromptPayload, temperature: float = 0.0) -> Tuple[str, Dict[str, int]]:
        """Generates response using GPT-5.1."""
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        client = OpenAI(api_key=self.api_key)
        messages = []

        # Few-shot examples
        if payload.few_shot_examples:
            for example_pdf_path, example_answer in payload.few_shot_examples:
                pdf_b64 = self._encode_pdf_to_base64(str(example_pdf_path))
                
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Extract data according to schema."},
                        {
                            "type": "input_file",
                            "filename": example_pdf_path.name,
                            "file_data": f"data:application/pdf;base64,{pdf_b64}"
                        }
                    ]
                })
                
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": example_answer}]
                })

        # Target PDF
        target_pdf_b64 = self._encode_pdf_to_base64(str(payload.target_pdf))
        messages.append({
            "role": "user",
            "content": [
                {"type": "input_text", "text": payload.instruction},
                {
                    "type": "input_file",
                    "filename": payload.target_pdf.name,
                    "file_data": f"data:application/pdf;base64,{target_pdf_b64}"
                }
            ]
        })

        # API call
        response = client.responses.create(
            model=self.model_version,
            input=messages,
            temperature=temperature
        )

        token_usage = {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens
        }
        
        return response.output_text, token_usage