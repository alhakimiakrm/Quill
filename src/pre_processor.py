import re
from collections import Counter

'''
This script "cleans" the text, tokenizes (or splits into words), and builds vocabulary
from those words by creating mapping from words to indices and vice versa.
The script then converts those tokens to sequences of indices based on the vocab, building, and sequence conversion steps.
load_corpus loads the text, which in this case is Hemingway's works.pre
'''

class PoemPreprocessor:
    def __init__(self, corpus):
        self.corpus = corpus
        self.vocab = None
        self.word_to_idx = None
        self.idx_to_word = None
        self.unk_token = "<UNK>"

    def clean_text(self, text):
        # Add space after punctuation if not already followed by a space
        text = re.sub(r'([?.!,"\'\)])(?! )', r'\1 ', text)
        # Remove space before punctuation
        text = re.sub(r'\s+([?.!,"\'\)])', r'\1', text)
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)  # Ensure no multiple spaces
        return text

    def tokenize_text(self, text):
        return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)

    def build_vocab(self, tokens):
        counter = Counter(tokens)
        self.vocab = [self.unk_token] + sorted(counter, key=counter.get, reverse=True)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}

    def text_to_sequences(self, tokens):
        return [self.word_to_idx.get(token, self.word_to_idx[self.unk_token]) for token in tokens]

    def preprocess(self):
        cleaned_corpus = self.clean_text(self.corpus)
        tokens = self.tokenize_text(cleaned_corpus)
        self.build_vocab(tokens)
        sequences = self.text_to_sequences(tokens)
        return sequences

def load_corpus(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def test_clean_text():
    preprocessor = PoemPreprocessor("")
    
    # Test cases
    test_cases = [
        ("Hello , world !", "hello, world!"),
        ("Hello,world!", "hello, world!"),
        ("Hello   world!", "hello world!"),
        ("Hello, world!", "hello, world!"),
        ("Hello...world!", "hello... world!"),
        ("Hello.  World!", "hello. world!"),
    ]
    
    for i, (input_text, expected_output) in enumerate(test_cases):
        output = preprocessor.clean_text(input_text)
        assert output == expected_output, f"Test case {i + 1} failed: {output} != {expected_output}"
        print(preprocessor.clean_text(input_text))
    print("All tests passed!")
    
    
test_clean_text()



