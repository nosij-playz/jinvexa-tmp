import json

from DataHandle.Utils.DataExtractor import DataExtract

extractor = DataExtract()

source = input("Enter URL or File Path: ")

result = extractor.extract(source)

print(json.dumps(result, indent=4, ensure_ascii=False))