from main import common_words, hway_aggregate, Counter, hway_theme
from cmnlist import common_words

def run_tests():
    hemingway_dir = 'Hemingway'
    aggregated_freq = hway_aggregate(hemingway_dir)
    assert aggregated_freq is not None, "Aggregated frequency dictionary is empty."
    for word in common_words:
        assert word in aggregated_freq, f"'{word}' not found in aggregated word frequencies."
    
    #print top 10 common words (this may not be very exemplerary of what I want to accomplish yet)
    most_common_words = Counter(aggregated_freq).most_common(10)
    print("\nMost common words in Hemingway's poems:")
    for word, freq in most_common_words:
        print(f"{word}: {freq}")

    #find the themes in hemginway's poems (this isn't fleshed out yet, as aforementioned)
    theme_words = ['home', 'love', 'war'] 
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
    

if __name__ == "__run_tests__":
        run_tests()