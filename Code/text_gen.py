import random
from collections import defaultdict

def generate_ngrams(words, n=3):
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(tuple(words[i:i + n]))
    return ngrams

def build_markov_model(ngrams):
    model = defaultdict(lambda: defaultdict(int))
    for *current_words, next_word in ngrams:
        model[tuple(current_words)][next_word] += 1

    for current_words, next_words in model.items():
        total_count = sum(next_words.values())
        for next_word in next_words:
            next_words[next_word] /= total_count
    return model

def generate_text(model, start_words, num_words=50):
    if not isinstance(start_words, tuple):
        start_words = tuple(start_words.split())
    story = list(start_words)
    current_words = start_words

    for _ in range(num_words):
        next_words = list(model[current_words].keys())
        next_word_weights = list(model[current_words].values())

        if not next_words:
            break

        next_word = random.choices(next_words, weights=next_word_weights)[0]
        story.append(next_word)
        current_words = tuple(story[-len(current_words):])

    return ' '.join(story)

def main():
    text = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."
    words = text.split()
    ngrams = generate_ngrams(words, 3)  # Using trigrams
    model = build_markov_model(ngrams)
    start_words = "The quick brown"
    generated_text = generate_text(model, start_words, 50)
    print(generated_text)

if __name__ == "__main__":
    main()
