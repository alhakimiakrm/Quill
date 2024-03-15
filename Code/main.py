import os
from lingua import Language, LanguageDetectorBuilder
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
import re

#build lingua once to recognize the following languages
detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.SPANISH, Language.GERMAN).build()

#read file 
def read_file(file_path): 
    try:
        with open(file_path, 'r', encoding='utf-8') as file: #encode in utf-8 format
            return file.read()
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return None

#iterate through TextSample and read each file
def read_all(directory_path): 
    texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            print(f"Processing file: {filename}") #DEBUG - REMOVE
            file_path = os.path.join(directory_path, filename)
            file_content = read_file(file_path)
            if file_content is not None:
                texts.append((filename, file_content))
    return texts

#check if language is in english
def is_english(text): 
    try:
        detected_language = detector.detect_language_of(text)
        return detected_language == Language.ENGLISH
    except Exception as e:
        print(f"Error during language detection: {e}")
        return False

#convert all txt files to a consistent format (lowercase, etc)
def preprocess(text_samples):
    text_samples = text_samples.lower()
    text_samples = re.sub(r'\s+', ' ', text_samples).strip()
    return text_samples
    
#parse all files in TextSamples and encode them in utf-8
def parse_all(directory):
    texts = []      #empty list for all texts
    all_words = []      #empty list for all words
    all_sentences = []      #empty list for all sentences
    for filename in os.listdir(directory):      #iterate through 'Hemingway
        if filename.endswith(".txt"):   #only read text files 
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:        #encode them in utf-8
                text = file.read()      #read each file
                processed = preprocess(text)        #process it (from preprocess above)
                texts.append(processed)         #add processed texts to 'texts'
                words = word_tokenize(processed)        #tokenize words list
                sentences = sent_tokenize(processed)    #tokenize all sentences
                all_words.extend(words)         #update words list
                all_sentences.extend(sentences)     #update sentences list
                print(f"{filename}: {len(words)} words, {len(sentences)} sentences") #DEBUG - REMOVE
    return texts, all_words, all_sentences
    
#main loop
def main():
    hemingway_dir = 'Hemingway'
    texts = read_all(hemingway_dir)

    if not texts:
        print(f"No text files found in {hemingway_dir}.")
        return

    eng_texts = []
    for filename, content in texts:
        print(f"Processing '{filename}'...", end=" ")
        if is_english(content):
            print(f"\033[92msuccess\033[0m")
            eng_texts.append((filename, content))
        else:
            print(f"\033[91mnot in English. Skipping..\033[0m]")

    if not eng_texts:
        print('No English texts were found. Please make sure the texts you upload are in English.')
    else:
        print(f"Processed {len(eng_texts)} English text files.")
        
    print(parse_all("Hemingway"))
    
    texts = read_all(hemingway_dir)
    print(f"Total files processed: {len(texts)}") #DEBUG - REMOVE
    
    if len(texts) != 14:
        print("Warning: Not all files were processed.") #DEBUG - REMOVE 
        
        
if __name__ == "__main__":
    main()
