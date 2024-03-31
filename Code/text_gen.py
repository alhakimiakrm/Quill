import random
from collections import defaultdict

# Example function to tokenize text into bigrams
def generate_bigrams(words):
    bigrams = []
    for i in range(len(words) - 1):
        bigrams.append((words[i], words[i + 1]))
    return bigrams

# Function to build the Markov chain model
def build_markov_model(bigrams):
    model = defaultdict(lambda: defaultdict(int))
    for current_word, next_word in bigrams:
        model[current_word][next_word] += 1
    # Convert counts to probabilities
    for current_word, next_words in model.items():
        total_count = sum(next_words.values())
        for next_word in next_words:
            next_words[next_word] /= total_count
    return model

# Function to generate text from the Markov model
def generate_text(model, start_word, num_words=50):
    current_word = start_word
    story = [current_word]
    for _ in range(num_words):
        next_words = list(model[current_word].keys())
        next_word_weights = list(model[current_word].values())
        
        if not next_words:
            break
        
        current_word = random.choices(next_words, weights=next_word_weights)[0]
        story.append(current_word)

    return ' '.join(story)

# Main function to tie everything together
def main():
    # Sample text for demonstration purposes
    text = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."
    words = text.split()

    # Generate bigrams from words
    bigrams = generate_bigrams(words)

    # Build the Markov model
    model = build_markov_model(bigrams)

    # Start generating text from a given word
    start_word = "The"  # Starting word to generate text
    generated_text = generate_text(model, start_word, 50)  # Generate 50 words of text

    print(generated_text)

if __name__ == "__main__":
    main()
