import re
from lingua import Language, LanguageDetectorBuilder
from nltk.tokenize import sent_tokenize, regexp_tokenize
from collections import defaultdict

# Build lingua once to recognize the specified languages
detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.SPANISH, Language.GERMAN).build()

# Function to read the file content
def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return None

# Function to check if the content is in English
def is_english(text):
    try:
        detected_language = detector.detect_language_of(text)
        return detected_language == Language.ENGLISH
    except Exception as e:
        print(f"Error during language detection: {e}")
        return False

# Preprocess the text
def preprocess(text_samples):
    text_samples = text_samples.lower() #convert all the text to lowercase
    
    # Handle contractions 
    contractions = {
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "won't": "will not", "can't": "cannot", "couldn't": "could not",
        "shouldn't": "should not", "wouldn't": "would not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
        "let's": "let us", "that's": "that is", "who's": "who is",
        "what's": "what is", "here's": "here is", "there's": "there is"
    }
    
    for contraction, full_form in contractions.items():
        text_samples = re.sub(r"\b{}\b".format(contraction), full_form, text_samples) 
    
    text_samples = re.sub(r'\s+', ' ', text_samples).strip()#replace whitespace chars with a single space
    text_samples = re.sub(r'[^\w\s]', '', text_samples) #standardize text, removing punctuation, symbols and anything else that is non-word
    
    return text_samples

# Tokenize the preprocessed text into words and sentences
def tokenize(text):
    words = regexp_tokenize(text, pattern=r'\b\w+\b')
    sentences = sent_tokenize(text)
    return words, sentences

# Calculate word frequencies in the text
def word_freq(text):
    freq = defaultdict(int)
    words = text.split()
    for word in words:
        if word.isalpha():
            freq[word] += 1
    return freq

# Main function
def main():
    file_path = 'Hemingway/hemingway1.txt'
    text_content = read_file(file_path)

    if text_content is None:
        print(f"Failed to read {file_path}.")
    elif is_english(text_content):
        print(f"\033[92mSuccessfully read {file_path}!\033[0m")
        return
    
    if not is_english(text_content): 
        print(f"\033[91mnot in English. Please adjust your text files and make sure they are in English.\033[0m]")
        return

    processed_text = preprocess(text_content)
    words, sentences = tokenize(processed_text)
    frequencies = word_freq(processed_text)

if __name__ == "__main__":
    main()
