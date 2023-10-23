import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

# URL of the webpage to analyze
webpageUrl = "https://www.cnn.com/2023/03/10/investing/svb-bank/index.html"

# Get the HTML content of the webpage
response = requests.get(webpageUrl)
htmlContent = response.text

# Parse the HTML content with BeautifulSoup
soup = BeautifulSoup(htmlContent, 'html.parser')

# Extract only the text of the article
articleText = ""
for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
    articleText += element.get_text()
#print(articleText)
# Perform sentiment analysis on the article text
blob = TextBlob(articleText)
sentiment = blob.sentiment.polarity

# Print the sentiment score
print("Sentiment score of the article: ", sentiment)
