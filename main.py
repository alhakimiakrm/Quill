import spacy
import nltk
from textblob import TextBlob

# Get user input
user_input = input("Enter your text: ")

# Create a TextBlob object
blob = TextBlob(user_input)

# Tokenize and tag parts of speech
for word, tag in blob.tags:
    print(word, tag)

# Analyze sentiment
sentiment = blob.sentiment
print("Polarity: ", sentiment.polarity)
print("Subjectivity: ", sentiment.subjectivity)
