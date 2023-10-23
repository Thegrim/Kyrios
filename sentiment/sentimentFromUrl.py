import requests
from textblob import TextBlob

# URL of the webpage to analyze
webpageUrl = "https://fortune.com/2023/03/09/bank-stocks-sink-silicon-valley-bank/"

# Get the text content of the webpage
response = requests.get(webpageUrl)
webpageText = response.text
print(webpageText)

# Perform sentiment analysis on the webpage's body of text
blob = TextBlob(webpageText)
sentiment = blob.sentiment.polarity

# Print the sentiment score
print("Sentiment score of the webpage: ", sentiment)
