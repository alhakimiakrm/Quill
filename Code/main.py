import os
from lingua import Language, LanguageDetectorBuilder

#build lingua once to recognize the following languages
detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.SPANISH, Language.GERMAN).build()

#read file 
def read_file(file_path): 
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return None

#itrate through TextSample and read each file
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
#main loop
def main():
    hemingway_dir = 'TextSamples'
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
