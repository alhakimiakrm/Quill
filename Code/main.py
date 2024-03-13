import os
import spacy
from lingua import Language, LanguageDetectorBuilder

detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.SPANISH, Language.GERMAN).build()

def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def read_all(directory_path):
    texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            file_content = read_file(file_path)
            if file_content is not None:
                texts.append((filename, file_content))
    return texts

def is_english(text):
    try:
        detected_language = detector.detect_language_of(text)
        return detected_language == Language.ENGLISH
    except Exception as e:
        print(f"Error during language detection: {e}")
        return False

def main():
    hemingway_dir = 'TextSamples'  
    texts = read_all(hemingway_dir)
 
    if not texts:
        print(f"No text files found in {hemingway_dir}.")
        return

    eng_texts = [text for text in texts if is_english(text[1])]
    for filename, _ in eng_texts:
        print(f"'{filename}' is in English. Processing...")

    if not eng_texts:
        print('No English texts were found. Please make sure the texts you upload are in English.')
    else:
        print(f"Processed {len(eng_texts)} English text files.")

if __name__ == "__main__":
    main()
