from __future__ import annotations

from typing import Any, Dict, Optional, List
from models import get_client, BaseLLMClient


class LLMClient:
    """
    Thin wrapper around provider SDKs so the pipeline can stay provider-agnostic.
    Now delegates to the 'models' package.
    """

    def __init__(
        self,
        model: str,
        provider: Optional[str] = None,  # kept for signature compatibility, though inferred internally
        temperature: float = 0.0,
    ):
        # We ignore 'provider' argument because the factory infers it from the model name,
        # or we could use it to select the client if we wanted to be more explicit.
        # But for now, get_client handles logic based on model name.
        self._delegate: BaseLLMClient = get_client(model, temperature)

    def generate(
        self,
        prompt: str,
        pdf_path: Optional[str] = None,
        pdf_paths: Optional[list[str]] = None,
        num_fewshot_pdfs: int = 0,
        **kwargs: Dict[str, Any],
    ) -> str:
        """
        Execute a text-only prompt and return the raw string output.
        """
        # Normalize pdf_paths
        if pdf_paths is None and pdf_path is not None:
            pdf_paths = [pdf_path]

        return self._delegate.generate(
            prompt=prompt,
            pdf_paths=pdf_paths,
            num_fewshot_pdfs=num_fewshot_pdfs,
            **kwargs
        )
