import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define the URL of the webpage to analyze
webpageUrl = "https://www.oaoa.com/local-news/good-news-united-donates-1000-pairs-of-socks-to-the-salvation-army/"

# Fetch the contents of the webpage
response = requests.get(webpageUrl)
soup = BeautifulSoup(response.content, 'html.parser')

# Extract the text from the webpage
#text = soup.get_text()
text = ""
for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
    text += element.get_text()

print(text)

# Load the pre-trained sentiment analysis model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Tokenize the text and convert it to a PyTorch tensor
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# Pass the input through the model to get the sentiment scores
outputs = model(**inputs)
scores = outputs.logits.detach().numpy()[0]

# Determine the most likely sentiment
sentiment = "positive" if scores.argmax() == 1 else "negative"

# Print the result
print(f"The sentiment of the webpage is {sentiment}.")
