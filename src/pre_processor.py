import re
from collections import Counter

class PoemPreprocessor:
    def __init__(self, corpus):
        self.corpus = corpus
        self.vocab = None
        self.word_to_idx = None
        self.idx_to_word = None

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower().strip()
        return text

    def tokenize_text(self, text):
        return text.split()

    def build_vocab(self, tokens):
        counter = Counter(tokens)
        self.vocab = sorted(counter, key=counter.get, reverse=True)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}

    def text_to_sequences(self, tokens):
        return [self.word_to_idx[token] for token in tokens]

    def preprocess(self):
        cleaned_corpus = self.clean_text(self.corpus)
        tokens = self.tokenize_text(cleaned_corpus)
        self.build_vocab(tokens)
        sequences = self.text_to_sequences(tokens)
        return sequences

def load_corpus(file_path):
    with open(file_path, 'r') as file:
        return file.read()