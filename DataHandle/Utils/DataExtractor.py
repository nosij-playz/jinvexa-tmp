import os
from urllib.parse import urlparse

from DataHandle.Utils.DocumentParser import DocumentParser
from DataHandle.Utils.ImageToText import ImageToText
from DataHandle.Utils.WebsiteParser import WebsiteParser
from DataHandle.Utils.YouTubeTranscript import YouTubeTranscript


class DataExtract:

    def __init__(self):

        self.youtube = YouTubeTranscript()
        self.document = DocumentParser()
        self.image = ImageToText()
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
            raise FileNotFoundError(source)

        ext = os.path.splitext(source)[1].lower()

        match ext:

            case ".pdf" | ".docx" | ".txt" | ".md" | ".py":

                return {
                    "type": "document",
                    "data": self.document.parse(source)
                }

            case ".png" | ".jpg" | ".jpeg" | ".bmp" | ".tif" | ".tiff" | ".webp":

                return {
                    "type": "image",
                    "data": self.image.extract(source)
                }

            case _:

                raise Exception(f"Unsupported input type: {ext}")