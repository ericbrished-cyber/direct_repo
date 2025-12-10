from typing import Tuple, Dict, List
from src.models.base import ModelAdapter
from src.prompts.builder import PromptPayload
import os
import base64
from openai import OpenAI

class GPTModel(ModelAdapter):
    """
    OpenAI GPT-5.1 implementation with native PDF support.
    Uses OpenAI's built-in PDF processing (text extraction + page images).
    """
    def __init__(self, model_version: str = "gpt-5.1"):
        self.model_version = model_version
        self.api_key = os.getenv("OPENAI_API_KEY")

    def _encode_pdf_to_base64(self, pdf_path: str) -> str:
        """
        Encodes a PDF file to base64 string.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        
        return base64.b64encode(pdf_bytes).decode("utf-8")

    def _create_user_message_with_pdf(self, text: str, pdf_path: str) -> Dict:
        """
        Constructs a user message with text and a PDF file.
        Uses OpenAI's native PDF input format.
        """
        pdf_b64 = self._encode_pdf_to_base64(pdf_path)
        
        content = [
            {
                "type": "input_text",
                "text": text
            },
            {
                "type": "input_file",
                "filename": os.path.basename(pdf_path),
                "file_data": f"data:application/pdf;base64,{pdf_b64}"
            }
        ]
        
        return {"role": "user", "content": content}
    
    def _create_assistant_message(self, text: str) -> Dict:
        """
        Constructs an assistant message.
        """
        return {
            "role": "assistant", 
            "content": [{"type": "output_text", "text": text}]
        }

    def generate(self, payload: PromptPayload, temperature: float = 0.0) -> Tuple[str, Dict[str, int]]:
        """
        Generates a response using GPT-5.1 with native PDF processing.
        
        GPT-5.1 automatically:
        - Extracts text from PDF
        - Creates images of each page
        - Can read both text AND figures/tables
        """
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        client = OpenAI(api_key=self.api_key, timeout=600.0)
        messages = []

        # Process Few-Shot Examples
        if payload.few_shot_examples:
            for example_pdf_path, example_answer in payload.few_shot_examples:
                try:
                    # Few-shot: PDF + instruction
                    messages.append(
                        self._create_user_message_with_pdf(
                            "Please extract the data according to the schema.",
                            str(example_pdf_path)
                        )
                    )
                    # Few-shot: Expected answer
                    messages.append(self._create_assistant_message(example_answer))
                except Exception as e:
                    print(f"Warning: Failed to process few-shot example {example_pdf_path}: {e}")

        # Process Target PDF
        messages.append(
            self._create_user_message_with_pdf(
                payload.instruction,
                str(payload.target_pdf)
            )
        )

        # Call API
        try:
            response = client.responses.create(
                model=self.model_version,
                input=messages,
                reasoning={"effort": "none"}, #must for temperature settings
                temperature=temperature
            )

            # Extract text
            raw_text = getattr(response, 'output_text', None)
            if raw_text is None and hasattr(response, 'choices'):
                raw_text = response.choices[0].message.content

            # Extract usage
            usage = getattr(response, 'usage', None)
            usage_dict = {}
            if usage:
                if hasattr(usage, 'model_dump'):
                    usage_dict = usage.model_dump()
                elif hasattr(usage, '__dict__'):
                    usage_dict = usage.__dict__

            token_usage = {
                "input": usage_dict.get("prompt_tokens", 0) if usage else 0,
                "output": usage_dict.get("completion_tokens", 0) if usage else 0
            }
            
            return raw_text, token_usage

        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")