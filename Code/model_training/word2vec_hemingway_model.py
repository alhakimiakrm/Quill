from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize 
import sys
import os
########
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent) 
########
from main import read_file, preprocess, read_all

#This is a sample script to begin training a model to find the 
#similarities in Hemingway's choice of vocabulary. 
#It works by first importing gensims 'Word2Vec', initializing the model (in w2vModel)
#and consequently calling functions from main to read and tokenize Hemingway's poems
#and returning the vector representation for each word. Words returned are vectors close to eachother 
#(based on the passed vocab example in this current case).

#Train a model to take 'sentences' and "learn" them. 
def w2vModel (sentences): 
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model


def main():
    dir = '../Hemingway'
    texts = read_all(dir) #read and process all poems in 'Hemingway' folder
    
    #This "pre-processes" all of the text based on a function from my main script, 
    # making it uniform so that it is easier to read. (remove punctuation, capitilization, etc.)
    #This also tokenizes the text into sentences
    sentences = []
    for _, text in texts:
        processed = preprocess(text)
        for sentence in sent_tokenize(processed):
            tokenized_sentence = word_tokenize(sentence.lower())
            sentences.append(tokenized_sentence)
    
    #Pass the sentences through 'w2vModel'
    model = w2vModel(sentences)
    model.save("hemingway.model") #This saves the model for future use elsewhere
    
    #This passes the word 'love' into the model, representing it as a vector and finding words that align closely
    #based off of its vector counterpart
    try:
        words = model.wv.most_similar('love', topn = 5)
        print(words)
    except KeyError:
        print('This word was not in found in the vocabulary') #Error handling 
    
if __name__ == "__main__":
    main()
    