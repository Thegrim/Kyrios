import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

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

# Assign a label to the article text
label = 'positive'

# Define a pipeline for sentiment analysis
pipeline = make_pipeline(
    CountVectorizer(),
    LogisticRegression()
)

# Fit the pipeline to the article text and label
pipeline.fit([articleText], [label])

# Predict the sentiment score of the article
sentimentScore = pipeline.predict_proba([articleText])[0][1]

# Print the sentiment score
print("Sentiment score of the article: ", sentimentScore)
