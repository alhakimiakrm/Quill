from spacy_langdetect import LanguageDetector
from spacy.language import Language
import spacy

@Language.factory("language_detector")
def create_lang_detector(nlp, name):
    return LanguageDetector()

nlp = spacy.blank("en")  # Create a blank English nlp object
nlp.add_pipe("sentencizer")  # Add sentencizer to the pipeline
nlp.add_pipe("language_detector", last=True)

doc = nlp("This is a test text file to examine if my language detector is kaput.")
print(doc._.language)
