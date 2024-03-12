import nltk
from textblob import TextBlob
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector


#load english model from spacy library
nlp = spacy.load("en_core_web_sm") 


def create_lang_detector(nlp, name):
    return LanguageDetector()

#add lang detector to pipeline
Language.factory("language_detector", func=create_lang_detector)
nlp.add_pipe("language_detector", last=True)

#test
sample_text = input('Enter text: ')
doc = nlp(sample_text)


def analyze_text(text):
    blob = TextBlob(text)
    return blob.sentiment

def read(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
hemingwaydir = os.path.join('..', TextSamples)


print(analyze_text(doc.text))
