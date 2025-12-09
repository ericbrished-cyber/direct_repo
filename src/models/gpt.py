from typing import Tuple, Dict, List
from src.models.base import ModelAdapter
from src.prompts.builder import PromptPayload
import os
import base64
import io
import pypdfium2 as pdfium
from PIL import Image
from openai import OpenAI

class GPTModel(ModelAdapter):
    """
    OpenAI GPT-5.1 implementation with Multimodal PDF support.
    """
    def __init__(self, model_version: str = "gpt-5.1"):
        self.model_version = model_version
        self.api_key = os.getenv("OPENAI_API_KEY")

    def _encode_pdf(self, pdf_path: str) -> List[str]:
        """
        Converts a PDF file to a list of base64 encoded JPEG images (one per page).
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        images_b64 = []
        try:
            pdf = pdfium.PdfDocument(pdf_path)
            for i in range(len(pdf)):
                page = pdf[i]
                # Render page to bitmap (scale=2.0 for better quality)
                bitmap = page.render(scale=2.0)
                pil_image = bitmap.to_pil()

                # Convert to base64
                buffered = io.BytesIO()
                pil_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                images_b64.append(img_str)
        except Exception as e:
            raise RuntimeError(f"Failed to process PDF {pdf_path}: {e}")

        return images_b64

    def _create_user_message(self, text: str, images: List[str]) -> Dict:
        """
        Constructs a user message with text and images.
        UPDATED: Uses 'input_text' and 'input_image' for newer API schema.
        """
        # ÄNDRING 1: "text" -> "input_text"
        content = [{"type": "input_text", "text": text}] 
        
        for img_b64 in images:
            content.append({
                # ÄNDRING 2: "image_url" -> "input_image"
                "type": "input_image", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}"
                }
            })
        return {"role": "user", "content": content}

    def _create_assistant_message(self, text: str) -> Dict:
        """
        Constructs an assistant message.
        """
        return {"role": "assistant", "content": text}

    def generate(self, payload: PromptPayload, temperature: float = 0.0) -> Tuple[str, Dict[str, int]]:
        """
        Generates a response using GPT-5.1.
        Uses client.responses.create with input=messages and reasoning={"effort": "none"}.
        """
        if not self.api_key:
             raise ValueError("OPENAI_API_KEY not found in environment variables.")

        client = OpenAI(api_key=self.api_key)
        messages = []

        # Process Few-Shot Examples
        if payload.few_shot_examples:
            for example_pdf_path, example_answer in payload.few_shot_examples:
                try:
                    example_images = self._encode_pdf(str(example_pdf_path))
                    # Add User message (PDF + generic instruction)
                    messages.append(self._create_user_message("Please extract the data according to the schema.", example_images))
                    # Add Assistant message (Expected Output)
                    messages.append(self._create_assistant_message(example_answer))
                except Exception as e:
                    print(f"Warning: Failed to process few-shot example {example_pdf_path}: {e}")

        # Process Target
        target_images = self._encode_pdf(str(payload.target_pdf))
        messages.append(self._create_user_message(payload.instruction, target_images))

        # Call API
        try:
            # Using client.responses.create. 'input' parameter takes the conversation history (messages).
            response = client.responses.create(
                model=self.model_version,
                input=messages,
                reasoning={"effort": "none"}, # Required for temperature support
                temperature=temperature
            )

            # Extract text
            raw_text = response.output_text

            # Extract usage (standard OpenAI fields: prompt_tokens, completion_tokens)
            usage = response.usage
            token_usage = {
                "input": usage.prompt_tokens if usage else 0,
                "output": usage.completion_tokens if usage else 0
            }

            return raw_text, token_usage

        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")
