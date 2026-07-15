import asyncio
import logging
import re
from typing import Dict, Any

from bs4 import BeautifulSoup
import trafilatura
from playwright.async_api import async_playwright

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebsiteParser:
    """
    A robust website parser that uses Playwright for rendering,
    BeautifulSoup for structure extraction, and Trafilatura for clean text.
    """

    def __init__(self):
        # Patterns to detect CAPTCHAs and bot verification pages
        self.verification_patterns = [
            r"g-recaptcha",
            r"hcaptcha",
            r"turnstile",
            r"cdn-cgi",
            r"cf-challenge",
            r"verify you are human",
            r"access denied",
            r"attention required",
            r"security check"
        ]

        # Tags to remove for cleaner extraction
        self.tags_to_remove = [
            "script", "style", "noscript", "svg", "iframe", "footer", "nav"
        ]

    def parse(self, url: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for the asynchronous parsing logic.
        Maintains compatibility with existing calls.
        """
        try:
            # Attempt to get the current event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # If a loop is already running, we can't use asyncio.run()
                # In a real production app, you'd use a library like nest_asyncio
                # or make the whole chain async. For this CLI tool, we'll try to
                # run it in a new thread or just use a synchronous wrapper.
                import threading
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._async_parse(url))
                    return future.result()
            else:
                return asyncio.run(self._async_parse(url))
        except Exception as e:
            logger.error(f"Critical error during parsing {url}: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def _async_parse(self, url: str) -> Dict[str, Any]:
        """
        The core asynchronous pipeline for parsing a website.
        """
        try:
            async with async_playwright() as p:
                # 1. Playwright: Render the page
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0 Safari/537.36"
                )
                page = await context.new_page()

                try:
                    # Navigate and wait until network is idle
                    await page.goto(url, wait_until="networkidle", timeout=30000)
                except Exception as e:
                    await browser.close()
                    return {"status": "error", "message": f"Navigation failed: {str(e)}"}

                html_content = await page.content()
                await browser.close()

                # 2. Verification Detection
                # Search for markers in the rendered HTML
                content_lower = html_content.lower()
                for pattern in self.verification_patterns:
                    if re.search(pattern, content_lower):
                        return {
                            "status": "verification_required",
                            "reason": "captcha",
                            "message": "Please upload a screenshot or PDF of this page."
                        }

                # 3. Trafilatura: Extract clean main content (full_text)
                # We use the rendered HTML content
                full_text = trafilatura.extract(html_content) or ""

                # 4. BeautifulSoup: Extract structured elements
                soup = BeautifulSoup(html_content, "html.parser")

                # Cleaning: Remove unwanted tags
                for tag in soup.find_all(self.tags_to_remove):
                    tag.decompose()

                # Further cleaning: Remove elements with ad-related classes or IDs
                for element in soup.find_all(True):
                    class_list = element.get("class", [])
                    id_val = element.get("id", "")
                    if any("ad-" in str(c).lower() or "sponsored" in str(c).lower() for c in class_list) or \
                       ("ad-" in id_val.lower() or "sponsored" in id_val.lower()):
                        element.decompose()

                # Extraction
                title = ""
                if soup.title:
                    title = soup.title.get_text(strip=True)
                elif soup.find("h1"):
                    title = soup.find("h1").get_text(strip=True)

                description = ""
                meta_desc = soup.find("meta", attrs={"name": "description"})
                if meta_desc:
                    description = meta_desc.get("content", "").strip()

                headings = []
                for h_tag in ["h1", "h2", "h3"]:
                    for h in soup.find_all(h_tag):
                        text = h.get_text(" ", strip=True)
                        if text:
                            headings.append(text)

                paragraphs = []
                for p in soup.find_all("p"):
                    text = p.get_text(" ", strip=True)
                    if text:
                        paragraphs.append(text)

                links = []
                for a in soup.find_all("a", href=True):
                    links.append({
                        "text": a.get_text(strip=True),
                        "url": a["href"]
                    })

                images = []
                for img in soup.find_all("img", src=True):
                    images.append({
                        "alt": img.get("alt", "").strip(),
                        "url": img["src"]
                    })

                tables = []
                for table in soup.find_all("table"):
                    table_data = []
                    rows = table.find_all("tr")
                    for row in rows:
                        cols = [col.get_text(" ", strip=True) for col in row.find_all(["td", "th"])]
                        if cols:
                            table_data.append(cols)
                    if table_data:
                        tables.append(table_data)

                # Metadata (buttons and forms)
                metadata = {
                    "buttons": [b.get_text(strip=True) for b in soup.find_all("button") if b.get_text(strip=True)],
                    "forms": len(soup.find_all("form"))
                }

                return {
                    "status": "success",
                    "title": title,
                    "description": description,
                    "headings": headings,
                    "paragraphs": paragraphs,
                    "links": links,
                    "images": images,
                    "tables": tables,
                    "metadata": metadata,
                    "full_text": full_text
                }

        except Exception as e:
            logger.error(f"Error in _async_parse for {url}: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
