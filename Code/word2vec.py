from gensim import Word2Vec
from main import all_sentences

model = Word2Vec(sentences=all_sentences, vector_size=100, window=5, min_count=1, workers=4)
model.save("hemingway_word2vec.model")