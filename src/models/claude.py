from typing import Tuple, Dict
from src.models.base import ModelAdapter
from src.prompts.builder import PromptPayload
import os
import base64
from anthropic import Anthropic

class ClaudeModel(ModelAdapter):
    """
    Anthropic Claude Opus 4.5 with prompt caching and PDF support.
    """
    def __init__(self, model_version: str = "claude-opus-4-5-20251101"):
        self.model_version = model_version
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

    def _encode_pdf_to_base64(self, pdf_path: str) -> str:
        """Encodes PDF to base64."""
        with open(pdf_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _create_document_block(self, pdf_path: str, use_cache: bool = False) -> Dict:
        """Creates a document content block with optional caching."""
        block = {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": self._encode_pdf_to_base64(str(pdf_path))
            }
        }
        
        if use_cache:
            block["cache_control"] = {"type": "ephemeral"}
        
        return block

    def generate(self, payload: PromptPayload) -> Tuple[str, Dict[str, int]]:
        """Generates response using Claude Opus 4.5 with prompt caching."""
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")

        client = Anthropic(api_key=self.api_key)
        messages = []

        # Few-shot examples - cache the last assistant response
        if payload.few_shot_examples:
            for idx, (example_pdf_path, example_answer) in enumerate(payload.few_shot_examples):
                is_last = (idx == len(payload.few_shot_examples) - 1)
                
                # User message with PDF (no cache)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract data according to schema."},
                        self._create_document_block(example_pdf_path, use_cache=False)
                    ]
                })
                
                # Assistant response - CACHE THE LAST ONE (this caches entire few-shot prefix)
                assistant_content = [{"type": "text", "text": example_answer}]
                if is_last:
                    assistant_content[0]["cache_control"] = {"type": "ephemeral"}
                
                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })

        # Target PDF - NO CACHE (each PDF is processed only once)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": payload.instruction},
                self._create_document_block(payload.target_pdf, use_cache=False)
            ]
        })

        # API call
        response = client.messages.create(
            model=self.model_version,
            max_tokens=4096,
            messages=messages,
            )

        # Extract text
        raw_text = ""
        for block in response.content:
            if block.type == "text":
                raw_text += block.text

        # Token usage
        token_usage = {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
            "cache_creation": getattr(response.usage, "cache_creation_input_tokens", 0),
            "cache_read": getattr(response.usage, "cache_read_input_tokens", 0)
        }
        
        return raw_text, token_usage