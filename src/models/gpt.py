from typing import Tuple, Dict
from src.models.base import ModelAdapter
from src.prompts.builder import PromptPayload
from src.models.dry_run import dump_debug_json, clean_gpt_messages
import os
import base64
from openai import OpenAI

class GPTModel(ModelAdapter):
    """
    OpenAI GPT-5.1 implementation with native PDF support.
    """
    def __init__(self, model_version: str = "gpt-5.2"):
        self.model_version = model_version
        self.api_key = os.getenv("OPENAI_API_KEY")

    def _encode_pdf_to_base64(self, pdf_path: str) -> str:
        """Encodes PDF to base64."""
        with open(pdf_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate(self, payload: PromptPayload, dry_run: bool = False) -> Tuple[str, Dict[str, int]]:
        """Generates response using GPT-5.1."""
        client = None if dry_run else OpenAI(api_key=self.api_key)
        messages = []

        # Few-shot examples
        if payload.few_shot_examples:
            for example in payload.few_shot_examples:
                example_pdf_path = example["pdf_path"]
                example_instruction = example["instruction"]
                example_answer = example["answer"]
                pdf_b64 = "<omitted>" if dry_run else self._encode_pdf_to_base64(str(example_pdf_path))
                
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": example_instruction},
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
        target_pdf_b64 = "<omitted>" if dry_run else self._encode_pdf_to_base64(str(payload.target_pdf))
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

        if dry_run:
            dump_debug_json("gpt_messages", clean_gpt_messages(messages))
            return "", {"input": 0, "output": 0}

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        # API call
        response = client.responses.create(
            model=self.model_version,
            reasoning={"effort": "medium"},
            input=messages
        )

        token_usage = {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens
        }
        
        return response.output_text, token_usage
