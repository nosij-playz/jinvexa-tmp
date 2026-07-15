from Utils.WebsiteParser import WebsiteParser

parser = WebsiteParser()

result = parser.parse(input("Enter URL: "))

print(result)