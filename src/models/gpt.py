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
                # ÄNDRING: Sänkt skala till 0.75 för att minska storleken drastiskt
                bitmap = page.render(scale=0.75)
                pil_image = bitmap.to_pil()

                # Convert to base64
                buffered = io.BytesIO()
                # ÄNDRING: Komprimerar bilden hårdare (quality=50) för snabbare upload
                pil_image.save(buffered, format="JPEG", quality=50, optimize=True)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                images_b64.append(img_str)
        except Exception as e:
            raise RuntimeError(f"Failed to process PDF {pdf_path}: {e}")

        return images_b64

    def _create_user_message(self, text: str, images: List[str]) -> Dict:
        """
        Constructs a user message with text and images.
        """
        content = [{"type": "input_text", "text": text}]
        
        for img_b64 in images:
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{img_b64}"
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
        """
        if not self.api_key:
             raise ValueError("OPENAI_API_KEY not found in environment variables.")

        # ÄNDRING: Sätter timeout till 5 minuter (300s) istället för default (ofta 60s)
        client = OpenAI(api_key=self.api_key, timeout=600.0)
        messages = []

        # Process Few-Shot Examples
        if payload.few_shot_examples:
            for example_pdf_path, example_answer in payload.few_shot_examples:
                try:
                    example_images = self._encode_pdf(str(example_pdf_path))
                    messages.append(self._create_user_message("Please extract the data according to the schema.", example_images))
                    messages.append(self._create_assistant_message(example_answer))
                except Exception as e:
                    print(f"Warning: Failed to process few-shot example {example_pdf_path}: {e}")

        # Process Target
        target_images = self._encode_pdf(str(payload.target_pdf))
        messages.append(self._create_user_message(payload.instruction, target_images))

        # Call API
        try:
            response = client.responses.create(
                model=self.model_version,
                input=messages,
                reasoning={"effort": "none"}, 
                temperature=temperature
            )

            # Extract text (Safe extraction)
            raw_text = getattr(response, 'output_text', None)
            if raw_text is None and hasattr(response, 'choices'):
                 raw_text = response.choices[0].message.content

            # Extract usage (Safe extraction)
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