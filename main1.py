import json

from Utils.DocumentParser import DocumentParser

parser = DocumentParser()

result = parser.parse("sample.pdf")

print(result["full_text"])

print(json.dumps(result, indent=4, ensure_ascii=False))