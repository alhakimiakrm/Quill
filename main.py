from src.train import train_model
from src.generate import generate_text

if __name__ == "__main__":
    corpus_path = "data/hemingway.txt"
    num_epochs = 20
    sequence_length = 10
    batch_size = 32
    learning_rate = 0.001

    # Train the model
    model, preprocessor = train_model(corpus_path, num_epochs, sequence_length, batch_size, learning_rate)

    # Generate text
    start_text = "soldiers never do die well"
    generated_text = generate_text(model, start_text, preprocessor, num_words=50, sequence_length=sequence_length)
    print(generated_text)
