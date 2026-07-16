# D:\Jinvexa\DataHandle\Utils\DataExtractor.py

import os
from urllib.parse import urlparse
from typing import Optional, Any

from DataHandle.Utils.DocumentParser import DocumentParser
from DataHandle.Utils.ImageToText import ImageToText
from DataHandle.Utils.WebsiteParser import WebsiteParser
from DataHandle.Utils.YouTubeTranscript import YouTubeTranscript


class DataExtract:

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize DataExtractor with an optional LLM client.
        
        Args:
            llm_client: The OllamaLLMClient instance from app.py
                       Used for image OCR and other AI-powered extractions
        """
        self.llm_client = llm_client
        
        # Initialize parsers with llm_client if needed
        self.youtube = YouTubeTranscript()
        self.document = DocumentParser()
        self.image = ImageToText(llm_client=llm_client)  # Pass llm_client to ImageToText
        self.website = WebsiteParser()

    def _is_url(self, value):
        try:
            p = urlparse(value)
            return p.scheme in ("http", "https")
        except Exception:
            return False

    def _is_youtube(self, url):
        hosts = [
            "youtube.com",
            "www.youtube.com",
            "youtu.be",
            "m.youtube.com"
        ]
        return any(host in url for host in hosts)

    def extract(self, source):
        """
        Extract data from the given source.
        
        Args:
            source: URL, file path, or YouTube link
            
        Returns:
            Dictionary with extracted data
        """
        # ---------------- URL ---------------- #
        if self._is_url(source):
            if self._is_youtube(source):
                return {
                    "type": "youtube",
                    "data": self.youtube.transcribe(source)
                }

            return {
                "type": "website",
                "data": self.website.parse(source)
            }

        # ---------------- FILE ---------------- #
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")

        ext = os.path.splitext(source)[1].lower()

        # Document files
        if ext in [".pdf", ".docx", ".txt", ".md", ".py"]:
            return {
                "type": "document",
                "data": self.document.parse(source)
            }

        # Image files
        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
            return {
                "type": "image",
                "data": self.image.extract(source)  # Uses llm_client for OCR
            }

        # Unsupported
        else:
            raise Exception(f"Unsupported input type: {ext}")