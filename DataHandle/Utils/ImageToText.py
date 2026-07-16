# D:\Jinvexa\DataHandle\Utils\ImageToText.py

import os
from typing import Optional, Any


class ImageToText:
    """
    Image to Text extraction using Ollama.
    Now receives the LLM client from DataExtractor.
    """

    def __init__(self, llm_client: Optional[Any] = None, model: Optional[str] = None):
        """
        Initialize ImageToText with either an LLM client or a model name.
        
        Args:
            llm_client: The OllamaLLMClient instance from app.py
            model: Model name (used if llm_client is not provided)
        """
        self.llm_client = llm_client
        
        # If no llm_client provided, use the model directly or fallback
        if self.llm_client:
            self.model = self.llm_client.model
        elif model:
            self.model = model
        else:
            # Fallback to default model from environment
            from dotenv import load_dotenv
            load_dotenv()
            self.model = os.getenv("OLLAMA_MODEL", "gemma4:latest")

    def extract(self, path: str) -> str:
        """
        Extract text from an image using Ollama.
        
        Args:
            path: Path to the image file
            
        Returns:
            Extracted text from the image
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        supported = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]
        
        if ext not in supported:
            raise ValueError(f"Unsupported image format: {ext}")

        # Use llm_client if available (preferred method)
        if self.llm_client and hasattr(self.llm_client, 'extract_text_from_image'):
            return self.llm_client.extract_text_from_image(path)
        
        # If llm_client doesn't have the method, use direct ollama
        import ollama
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an OCR engine.\n"
                            "Extract every visible piece of text exactly as written.\n"
                            "Do not summarize.\n"
                            "Do not explain.\n"
                            "Do not translate.\n"
                            "Do not correct spelling or grammar.\n"
                            "Preserve line breaks, headings, tables, bullet points, and spacing as closely as possible.\n"
                            "If no text is present, return exactly: NO_TEXT_FOUND"
                        )
                    },
                    {
                        "role": "user",
                        "content": "Extract all text from this image.",
                        "images": [path]
                    }
                ]
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"❌ Image extraction error: {e}")
            return "NO_TEXT_FOUND"