from Utils.YouTubeTranscript import YouTubeTranscript

yt = YouTubeTranscript()

url = input("Enter YouTube URL: ")

result = yt.transcribe(url)

print(result["full_text"])