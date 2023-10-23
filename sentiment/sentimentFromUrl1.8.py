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
text = ""
for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
    text += element.get_text()

# Load the pre-trained sentiment analysis model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Tokenize the text and convert it to a PyTorch tensor
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# Pass the input through the model to get the sentiment scores
outputs = model(**inputs)
scores = outputs.logits.detach().numpy()[0]

# Calculate the sentiment score between -2 and 2
sentiment_score = (scores[1] - scores[0]) * 2

# Print the result
print(f"The sentiment score of the webpage is {sentiment_score}.")
