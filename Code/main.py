import os
from lingua import Language, LanguageDetectorBuilder
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
from collections import defaultdict, Counter
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

#convert all txt files to a consistent format (lowercase, remove punctuation etc.)
def preprocess(text_samples):
    text_samples = re.sub(r'[ \t]+', ' ', text_samples)
    text_samples = re.sub(r'[^\w\s\.\,\?\!\-]', '', text_samples)
    return text_samples
    
#parse all files and tokenize words and sentences
def parse_all(directory):
    texts = []      
    all_words = []      
    all_sentences = []      
    for filename in os.listdir(directory):      #iterate through 'Hemingway
        if filename.endswith(".txt"):   #only read text files 
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:   #encode them in utf-8
                text = file.read()   
                processed = preprocess(text)   
                texts.append(processed)       
                words = regexp_tokenize(processed, pattern=r'\b\w+\b')   #return whole words, ignoring punctuation or whitespace 
                sentences = sent_tokenize(processed)    
                all_words.extend(words)       
                all_sentences.extend(sentences)   
    return texts, all_words, all_sentences


#measure hemingway's word frequency 
def hway_words(poem):
    word_freq = defaultdict(int)
    words = word_tokenize(poem.lower())
    for word in words:
        if word.isalpha():
            word_freq[word] += 1
    return word_freq

#create a cluster of hemingways common words
def hway_aggregate(hemingway):
    aggregated_freq = defaultdict(int)
    
    #this iterates through the directory, and begins to find the frequency of words in hemingways poems
    for filename in os.listdir(hemingway):
        if filename.endswith (".txt"):
            file_path = os.path.join(hemingway, filename)
            with open (file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                word_freq = hway_words(text)
                
                for word, count in word_freq.items():
                    aggregated_freq[word] += count
    return aggregated_freq

#find hemingways themes #TODO Expand this to better capture the themes of Hemingway's poems 
def hway_theme(hway_aggregate, theme_words):
    theme_freq = {word: hway_aggregate[word] for word in theme_words if word in hway_aggregate}
    return theme_freq

#test the aggregate, theme and word frequency functions 
def run_tests():
    hemingway_dir = 'Hemingway'
    aggregated_freq = hway_aggregate(hemingway_dir)
    assert aggregated_freq is not None, "Aggregated frequency dictionary is empty."
    common_words = ['the', 'and', 'of', 'sing', 'age', 'away','must', 'horned', 'owl', 'youth', 'boy', 'boy', 'hotel']  #just a few test words
    for word in common_words:
        assert word in aggregated_freq, f"'{word}' not found in aggregated word frequencies."
    
    #print top 10 common words (this may not be very exemplerary of what I want to accomplish yet)
    most_common_words = Counter(aggregated_freq).most_common(10)
    print("\nMost common words in Hemingway's poems:")
    for word, freq in most_common_words:
        print(f"{word}: {freq}")

    #find the themes in hemginway's poems (this isn't fleshed out yet, as aforementioned)
    theme_words = ['love', 'death', 'war', 'home', 'shit', 'art', 'dear',  ] 
    theme_freq = hway_theme(aggregated_freq, theme_words)
    assert theme_freq, "No theme words found in Hemingway's poems."
    
    
    print("\nTheme word frequencies in Hemingway's poems:")
    for word in theme_words:
        if word in theme_freq:
            print(f"{word}: {theme_freq[word]}")
        else:
            print(f"{word}: not found")
    
    for word in theme_words:
        assert word in theme_freq, f"Theme word '{word}' not found in Hemingway's themes."

    print("All tests passed!")
    
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
    run_tests()