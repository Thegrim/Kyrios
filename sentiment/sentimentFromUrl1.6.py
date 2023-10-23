import requests
from bs4 import BeautifulSoup
import torch
import torch.nn as nn
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Define the sentiment analysis model
class SentimentAnalysisModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

# Load the pre-trained sentiment analysis model
model_path = "sentiment_analysis_model.pth"
model = SentimentAnalysisModel(768, 256, 2)
model.load_state_dict(torch.load(model_path))
model.eval()

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

# Convert the article text to a PyTorch tensor
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
articleTokens = tokenizer.encode(articleText, add_special_tokens=True)
articleTensor = torch.tensor(articleTokens)

# Perform sentiment analysis on the article text using the PyTorch model
with torch.no_grad():
    output = model(articleTensor.unsqueeze(0))
    scores = output.squeeze().numpy()

# Print the sentiment score
print("Sentiment score of the article: ", scores[1] - scores[0])
