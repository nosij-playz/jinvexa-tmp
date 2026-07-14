from Utils.ImageToText import ImageToText

ocr = ImageToText()

text = ocr.extract("IMG-20251013-WA0003.jpg")

print(text)