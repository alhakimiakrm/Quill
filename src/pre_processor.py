import re
from collections import Counter

class PoemPreprocessor:
    def __init__(self, corpus, user_input=""):
        self.corpus = corpus + " " + user_input
        self.vocab = None
        self.word_to_idx = None
        self.idx_to_word = None
        self.unk_token = "<UNK>"

    def clean_text(self, text):
        # Normalize line breaks and remove extra whitespace
        text = re.sub(r'\r\n|\r|\n', ' ', text)  # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)  # Ensure no multiple spaces

        # Add space after punctuation if not already followed by a space
        text = re.sub(r'([?.!,"\'\)])(?! )', r'\1 ', text)
        # Remove space before punctuation
        text = re.sub(r'\s+([?.!,"\'\)])', r'\1', text)
        # Convert text to lowercase
        text = text.lower().strip()
        
        # Replace dashes/hyphens with a single space
        text = re.sub(r'\s*[-–—]+\s*', ' ', text)

        # Remove any leading or trailing spaces
        text = text.strip()

        return text

    def tokenize_text(self, text):
        # Tokenize based on words and punctuation
        return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)

    def build_vocab(self, tokens):
        # Build vocabulary with a counter
        counter = Counter(tokens)
        self.vocab = [self.unk_token] + sorted(counter, key=counter.get, reverse=True)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}

    def text_to_sequences(self, tokens):
        # Convert tokens to sequences
        return [self.word_to_idx.get(token, self.word_to_idx[self.unk_token]) for token in tokens]

    def preprocess(self):
        # Full preprocessing pipeline
        cleaned_corpus = self.clean_text(self.corpus)
        tokens = self.tokenize_text(cleaned_corpus)
        self.build_vocab(tokens)
        sequences = self.text_to_sequences(tokens)
        return sequences

def load_corpus(file_path):
    with open(file_path, 'r') as file:
        return file.read()

