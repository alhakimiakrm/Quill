import os
import spacy
import langdetect
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from spacy.tokens import Doc

if not Doc.has_extension("lang"): 
    Doc.set_extension("lang", default=None)

nlp = spacy.load("en_core_web_sm") #loading english language model from spacy

@Language.factory("language_detector")
def create_lang_detector(nlp, name): #language detector
    return LanguageDetector()

nlp.add_pipe("language_detector", last=True) #adding detector to pipeline

def read_file(file_path): #this allows us to read a txt file in 'TextSamples'
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_all(directory_path): #reads all 3 files at once
    texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            texts.append((filename, read_file(file_path)))
    return texts

def is_english(text): #check to confirm the texts are in english 
    doc = nlp(text)
    print(f"Detected language: {doc._.language['language']}, Score: {doc._.language['score']}")
    return doc._.lang == 'en'


def main():
    hemingway_dir = 'TextSamples'
    texts = read_all(hemingway_dir)
    
    #if the texts are not being read or there are no texts...
    if not texts:
        print(f"No text files found in {hemingway_dir}.")
        return
    
    #this checks that all text files are english and rejects them if they are not
    eng_texts = 0
    for filename, text in texts:
        if is_english(text):
            print(f"'{filename}' is in English. Processing...")
            print(f"Read '{filename}' successfully.")
            eng_texts += 1
        else:
            print(f"'{filename}' is not in English. Skipping...")
        
    if all(not is_english(text) for _, text in texts):
        print('No English texts were found. Please make sure the texts you upload are in English.')  
        
        
    print(nlp.pipe_names)
        
if __name__ == "__main__": #main code
   main()