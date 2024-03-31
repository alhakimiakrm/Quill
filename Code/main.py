import os
from lingua import Language, LanguageDetectorBuilder
from nltk.tokenize import sent_tokenize, regexp_tokenize
from collections import defaultdict
import re


#build lingua once to recognize the following languages
detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.SPANISH, Language.GERMAN).build()

#read file 
def read_file(file_path): 
    try:
        with open(file_path, 'r', encoding='utf-8') as file: #encode in utf-8 format
            return file.read()
    except IOError as e:
        print(f"Error reading file {file_path}: {e}") #error handling
        return None

#iterate through TextSample and read each file
def read_all(directory_path): 
    texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"): 
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

#convert all txt files to a consistent format (lowercase, remove punctuation etc.
def preprocess(text_samples):
    text_samples = re.sub(r'\s+', ' ', text_samples)
    text_samples = re.sub(r'[^\w\s-]', '', text_samples)
    text_samples = text_samples.lower()
    return text_samples

    
#parse all files and tokenize words and sentences
def parse_all(Hemingway):
    texts = []      
    all_words = []      
    all_sentences = []      
    for filename in os.listdir(Hemingway):      #iterate through 'Hemingway
        if filename.endswith(".txt"):   #only read text files 
            with open(os.path.join(Hemingway, filename), 'r', encoding='utf-8') as file:   #encode them in utf-8
                text = file.read()   
                processed = preprocess(text)   
                texts.append(processed)       
                words = regexp_tokenize(processed, pattern=r'\b\w+\b')   #return whole words, ignoring punctuation or whitespace 
                sentences = sent_tokenize(processed)    
                all_words.extend(words)       
                all_sentences.extend(sentences) 
                print(sentences)  
    return texts, all_words, all_sentences


def wordFreq(Hemingway):
    combined_freq = defaultdict(int)
    
    for filename in os.listdir(Hemingway):
        if filename.endswith(".txt"):
            file_path = os.path.join(Hemingway, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    processed_text = preprocess(text)  
                    words = processed_text.split() 
                    
                    for word in words:
                        if word.isalpha():  # Check if the token is a word
                            combined_freq[word] += 1

            except IOError as e:
                print(f"Error reading file {file_path}: {e}")
    return combined_freq
                    
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
        
if __name__ == "__main__":
    main()