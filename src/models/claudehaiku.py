from typing import Tuple, Dict
from src.models.base import ModelAdapter
from src.prompts.builder import PromptPayload
from src.models.dry_run import dump_debug_json, clean_claude_messages
import os
import base64
from anthropic import Anthropic

class ClaudeHaikuModel(ModelAdapter):
    """
    Anthropic Claude Haiku 4.5 with prompt caching and extended thinking.
    Uses native Anthropic API with thinking budget for reasoning control.
    """
    def __init__(self, model_version: str = "claude-haiku-4-5-20251001"):
        self.model_version = model_version
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        # Thinking budget: 10000 = balanced, increase for complex tasks
        self.thinking_budget = int(os.getenv("CLAUDE_HAIKU_THINKING_BUDGET", "10000"))

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

    def generate(self, payload: PromptPayload, dry_run: bool = False) -> Tuple[str, Dict[str, int]]:
        """Generates response using Claude Haiku 4.5 with extended thinking."""
        client = None if dry_run else Anthropic(api_key=self.api_key)
        messages = []

        # Few-shot examples: cache only the final assistant response
        if payload.few_shot_examples:
            for idx, example in enumerate(payload.few_shot_examples):
                is_last = (idx == len(payload.few_shot_examples) - 1)
                example_pdf_path = example["pdf_path"]
                example_instruction = example["instruction"]
                example_answer = example["answer"]
                
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": example_instruction}, 
                        self._create_document_block(example_pdf_path, use_cache=False)
                    ]
                })
                
                assistant_content = [{"type": "text", "text": example_answer}]
                if is_last:
                    assistant_content[0]["cache_control"] = {"type": "ephemeral"}
                
                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })

        # Target PDF
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": payload.instruction},
                self._create_document_block(payload.target_pdf, use_cache=False)
            ]
        })

        if dry_run:
            dump_debug_json("claude_haiku_messages", clean_claude_messages(messages))
            return "", {"input": 0, "output": 0, "cache_creation": 0, "cache_read": 0}

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")

        # API call with extended thinking
        response = client.messages.create(
            model=self.model_version,
            max_tokens=8192,
            messages=messages,
            thinking={
                "type": "enabled",
                "budget_tokens": self.thinking_budget
            }
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
            "cache_read": getattr(response.usage, "cache_read_input_tokens", 0),
            "thinking": getattr(response.usage, "thinking_tokens", 0)
        }
        
        return raw_text, token_usage
