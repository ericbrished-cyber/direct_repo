from typing import Tuple, Dict, List
from src.models.base import ModelAdapter
from src.prompts.builder import PromptPayload
from src.models.dry_run import dump_debug_json
import os
import base64
from google import genai
from google.genai import types

class GeminiModel(ModelAdapter):
    """
    Google Gemini 3 Pro implementation with native PDF support.
    Uses Gemini's built-in PDF processing and reasoning capabilities.
    """
    def __init__(self, model_version: str = "gemini-3-pro-preview"):
        self.model_version = model_version
        self.api_key = os.getenv("GOOGLE_API_KEY")

    def _encode_pdf_to_base64(self, pdf_path: str) -> bytes:
        """
        Reads PDF file as raw bytes for inline blob.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        with open(pdf_path, "rb") as f:
            return f.read()

    def _create_content_with_pdf(self, text: str, pdf_path: str) -> types.Content:
        """
        Creates a Content object with text and PDF inline blob.
        """
        pdf_bytes = self._encode_pdf_to_base64(pdf_path)
        
        return types.Content(
            role="user",
            parts=[
                types.Part(text=text),
                types.Part(
                    inline_data=types.Blob(
                        mime_type="application/pdf",
                        data=pdf_bytes
                    )                
                )
            ]
        )

    def _create_model_response(self, text: str) -> types.Content:
        """
        Creates a model (assistant) response for few-shot examples.
        """
        return types.Content(
            role="model",
            parts=[types.Part(text=text)]
        )

    def generate(self, payload: PromptPayload, dry_run: bool = False) -> Tuple[str, Dict[str, int]]:
        """
        Generates a response using Gemini 3 Pro with native PDF processing.
        
        Args:
            payload: Prompt payload with PDFs and instructions
                    
        Note: Uses 'high' thinking level by default for maximum reasoning depth.
        """
        contents = []

        # Process Few-Shot Examples
        if payload.few_shot_examples:
            for example in payload.few_shot_examples:
                example_pdf_path = example["pdf_path"]
                example_instruction = example["instruction"]
                example_answer = example["answer"]
                if dry_run:
                    contents.append({
                        "role": "user",
                        "text": example_instruction,
                        "pdf": os.path.basename(example_pdf_path)
                    })
                    contents.append({"role": "model", "text": example_answer})
                else:
                    try:
                        # User: PDF + instruction
                        contents.append(
                            self._create_content_with_pdf(
                                example_instruction,
                                str(example_pdf_path)
                            )
                        )
                        # Model: Expected answer
                        contents.append(self._create_model_response(example_answer))
                    except Exception as e:
                        print(f"Warning: Failed to process few-shot example {example_pdf_path}: {e}")

        # Process Target PDF
        if dry_run:
            contents.append({
                "role": "user",
                "text": payload.instruction,
                "pdf": os.path.basename(payload.target_pdf)
            })
        else:
            contents.append(
                self._create_content_with_pdf(
                    payload.instruction,
                    str(payload.target_pdf)
                )
            )

        if dry_run:
            dump_debug_json("gemini_contents", contents)
            return "", {"input": 0, "output": 0}

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")

        client = genai.Client(api_key=self.api_key)

        # Call API with reasoning configuration
        try:
            response = client.models.generate_content(
                model = self.model_version,
                contents = contents,
                config = types.GenerateContentConfig(
                    thinking_config = types.ThinkingConfig(
                        thinking_level = "high"
                    )
                )
            )

            # Extract text
            raw_text = response.text if hasattr(response, 'text') else ""
            
            # Extract usage metadata
            usage = getattr(response, 'usage_metadata', None)
            token_usage = {
                "input": 0,
                "output": 0
            }
            
            if usage:
                # Gemini uses different field names
                token_usage["input"] = getattr(usage, 'prompt_token_count', 0)
                token_usage["output"] = getattr(usage, 'candidates_token_count', 0)

            return raw_text, token_usage

        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}")
