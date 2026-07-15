import os

import ollama

from Config.Config import Config


class ImageToText:

    def __init__(self):

        self.config = Config()

        self.model = self.config.get_model()

    def extract(self, path):

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        ext = os.path.splitext(path)[1].lower()

        if ext not in [
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp"
        ]:
            raise Exception("Unsupported image format")

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